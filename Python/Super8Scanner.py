# Super8Scanner.py
#
# (c)2021 Stuart Pittaway
#
#
import numpy as np
import cv2 as cv
import os
import serial
from serial.serialwin32 import Serial
import serial.tools.list_ports as port_list
from datetime import datetime
from time import sleep


def pointInRect(point, rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False


def MarlinWaitForReply(MarlinSerialPort: Serial, echoToPrint=True) -> bool:
    tstart = datetime.now()

    while True:
        # Wait until there is data waiting in the serial buffer
        if MarlinSerialPort.in_waiting > 0:
            # Read data out of the buffer until a CR/NL is found
            serialString = MarlinSerialPort.readline()

            if echoToPrint:
                if serialString.startswith(b"echo:"):
                    # Print the contents of the serial data
                    print("Marlin R:", serialString.decode("Ascii"))

            if serialString == b"ok\n":
                return True

            # Reset delay since last reception
            tstart = datetime.now()

        else:
            # Abort after X seconds of not receiving anything
            duration = datetime.now()-tstart
            if duration.total_seconds() > 3:
                return False


def SendMarlinCmd(MarlinSerialPort: Serial, cmd: str) -> bool:
    #print("Sending GCODE",cmd)

    if MarlinSerialPort.isOpen() == False:
        raise Exception("Port closed")

    # Flush input buffer
    MarlinSerialPort.flushInput()
    MarlinSerialPort.flushOutput()
    MarlinSerialPort.read_all()

    MarlinSerialPort.write(cmd.encode('utf-8'))
    MarlinSerialPort.write(b'\n')
    if MarlinWaitForReply(MarlinSerialPort) == False:
        raise Exception("Bad GCODE command or not a valid reply from Marlin")

    return True


def SendMultipleMarlinCmd(MarlinSerialPort: Serial, cmds: list) -> bool:
    for cmd in cmds:
        SendMarlinCmd(MarlinSerialPort, cmd)
    return True


def ProcessImage(videoCaptureObject, centre_box: list, video_width: int, video_height: int):
    cap, frame = videoCaptureObject.read()

    if cap == False:
        raise IOError("Failed to capture image")

    # Capture of 1280x720 image is 1280 x 720 x 3/2 (1382400 bytes)
    # 1280x720 16 bit
    # ImgSz=1843200
    # 1843200 Sample Size
    # SubType YUY2
    shape = (int(video_height * 1.5), int(video_width))
    frame = frame.reshape(shape)
    # Convert YUV2 into RGB for OpenCV to use
    frame = cv.cvtColor(frame, cv.COLOR_YUV2BGR_NV12)

    # Mirror horizontal - sproket is now on left of image
    frame = cv.flip(frame, 1)

    # Normally 1280x720 for 1M camera
    image_width = frame.shape[:2][1]
    image_height = frame.shape[:2][0]

    # Mask out any plastic gate on left/right/top/bottom of frame
    gate_mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv.rectangle(gate_mask, (75, 0), (image_width-75, image_height), 255, -1)
    frame = cv.bitwise_and(frame, frame, mask=gate_mask)

    # Mask left side of image to find the sprokets, crop out words on film like "KODAK LABS"
    # we are looking for a narrow vertical section of the sprokets, not including any film picture
    # or the curved corners of the sproket hole.
    sproket_mask = np.zeros(frame.shape[:2], dtype="uint8")
    x = int(image_width*0.15)
    cv.rectangle(sproket_mask, (130, 0), (x, image_height), 255, -1)
    masked = cv.bitwise_and(frame, frame, mask=sproket_mask)
    # cv.imshow("masked",masked)

    # Blur the image and convert to grayscale
    matrix = (17, 7)
    frame_blur = cv.GaussianBlur(masked, matrix, 0)
    imgGry = cv.cvtColor(frame_blur, cv.COLOR_BGR2GRAY)

    # Threshold to only keep the sproket data visible (bright white now)
    _, thrash = cv.threshold(imgGry, 200, 255, cv.THRESH_BINARY)
    # cv.imshow("thrash",thrash)

    # find Canny Edges
    canny_edges = cv.Canny(thrash, 30, 200)
    # cv.imshow("canny_edges",canny_edges)

    # Get contour of the sproket
    contours, _ = cv.findContours(
        canny_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Draw the target centre box we are looking for (just for debug)
    cv.rectangle(frame, (centre_box[0], centre_box[1]), (
        centre_box[0]+centre_box[2], centre_box[1]+centre_box[3]), (0, 255, 0), 2)

    # Sort by area, largest first (hopefully our sproket - we should only have 1 full sprocket in view at any 1 time)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    if len(contours) > 0:
        # Just take the first one...
        contour = contours[0]

        # Find area of detected shapes and filter on the larger ones
        area = cv.contourArea(contour)

        # Sproket must be bigger than this to be okay...
        if area > 8500:
            # (center(x, y), (width, height), angleofrotation) = cv.minAreaRect(contour)
            rect = cv.minAreaRect(contour)
            rotation = rect[2]
            # Gets center of rotated rectangle
            box = cv.boxPoints(rect)
            box = np.int0(box)
            centre = rect[0]
            colour = (0, 0, 255)

            # Mark centre of sproket with a circle
            cv.circle(frame, (int(centre[0]), int(
                centre[1])), 8, (0, 100, 100), -1)

            # Draw the rectangle
            cv.drawContours(frame, [box], 0, colour, 2)
            return frame, centre
        else:
            #print("Area is ",area)
            pass
    else:
        cv.putText(frame, "No contour", (0, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    return frame, None


def MoveFilm(marlin: Serial, y: float, feed_rate: int):
    SendMarlinCmd(marlin, "G0 Y{0:.4f} F{1}".format(y, feed_rate))
    # Dwell
    #SendMarlinCmd(marlin,"G4 P100")
    # Wait for move complete
    SendMarlinCmd(marlin, "M400")


def ConnectToMarlin():
    #ports = list(port_list.comports())
    # for p in ports:
    #    print (p)

    # Connect to MARLIN
    marlin = serial.Serial(
        port="COM5", baudrate=250000, bytesize=8, timeout=5, stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE
    )

    # After initial connection Marlin sends loads of information which we ignore...
    MarlinWaitForReply(marlin, False)

    # Send setup commands...
    # M502 Hardcoded Default Settings Loaded
    # G21 - Millimeter Units
    # M211 - Software Endstops (disable)
    # G90 - Absolute Positioning
    # M106 - Fan On (LED LIGHT)
    # G92 - Set Position
    # M201 - Set Print Max Acceleration (off)
    SendMultipleMarlinCmd(
        marlin, ["M502", "G21", "M211 S0", "G90", "M106", "G92 X0 Y0 Z0", "M201 Y0"])

    # M92 - Set Axis Steps-per-unit
    # Steps per millimeter - 51.5 for 19mm rubber band wheel (lumpy and not circle!)
    # Just a fake number!
    SendMarlinCmd(marlin, "M92 Y10")

    # Wait for movement to complete
    SendMarlinCmd(marlin, "M400")

    return marlin


def DisconnectFromMarlin(serial_port: Serial):
    # M107 Light Off
    # M84 Steppers Off
    SendMultipleMarlinCmd(serial_port, ["M107", "M84"])
    serial_port.close()


def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def ConfigureCamera():
    # Open webcamera
    videoCaptureObject = cv.VideoCapture(1, cv.CAP_MSMF)

    if not (videoCaptureObject.isOpened()):
        raise Exception("Could not open video device")

    if videoCaptureObject.set(cv.CAP_PROP_FRAME_WIDTH, 1280) == False:
        raise Exception("Unable to set video capture parameter")

    if videoCaptureObject.set(cv.CAP_PROP_FRAME_HEIGHT, 720) == False:
        raise Exception("Unable to set video capture parameter")

    # Request raw camera data (YUV mode, to avoid JPEG artifacts)
    if videoCaptureObject.set(cv.CAP_PROP_CONVERT_RGB, 0) == False:
        raise Exception("Unable to set video capture parameter")

    if decode_fourcc(videoCaptureObject.get(cv.CAP_PROP_FOURCC)) != "NV12":
        raise Exception("Camera not in raw YUV4:2:0 format")

    video_width = videoCaptureObject.get(cv.CAP_PROP_FRAME_WIDTH)
    video_height = videoCaptureObject.get(cv.CAP_PROP_FRAME_HEIGHT)

    return videoCaptureObject, video_width, video_height


def OutputFolder() -> str:
    # Image Output path - create if needed
    path = os.path.join(os.getcwd(), "Capture")

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def StartupAlignment(marlin: Serial, videoCaptureObject, centre_box, video_width, video_height):
    marlin_y = 0

    return_value = False

    while True:
        my_frame, centre = ProcessImage(
            videoCaptureObject, centre_box, video_width, video_height)

        if centre == None:
            cv.putText(my_frame, "Sproket not detected, press f to nudge forward, b for back, j to jump forward quickly, ESC to quit",
                       (0, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv.LINE_AA)
        else:
            cv.putText(my_frame, "Sproket detected, press SPACE to start scanning, ESC to quit",
                       (0, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow('RawVideo', my_frame)

        # Check keyboard
        k = cv.waitKey(10) & 0xFF

        if k == ord(' '):    # SPACE key to continue
            return_value = True
            break

        if k == 27:
            return_value = False
            break

        if k == ord('f'):
            marlin_y += 1
            MoveFilm(marlin, marlin_y, 500)

        if k == ord('j'):
            marlin_y += 100
            MoveFilm(marlin, marlin_y, 8000)

        if k == ord('b'):
            marlin_y -= 1
            MoveFilm(marlin, marlin_y, 500)

    return return_value


def main():
    #print( cv.__version__ )

    path = OutputFolder()
    marlin = ConnectToMarlin()
    videoCaptureObject, video_width, video_height = ConfigureCamera()

    # Constants (sort of)
    NUDGE_FEED_RATE = 320
    STANDARD_FEED_RATE = 5000

    # This is the trigger rectangle for the sproket identification
    # must be in the centre of the screen without cropping each frame of Super8
    # A frame is W1280 and H720 (based on input web camera)

    centre_box = [130, 0, 50, 40]
    centre_box[1] = int(video_height/2-centre_box[3]/2)

    if StartupAlignment(marlin, videoCaptureObject, centre_box, video_width, video_height) == True:

        # Total number of images stored as a unique frame
        frame_number = 0
        # Position on film reel (in marlin Y units)
        marlin_y = 0.0
        # Total number of images taken
        frame_counter = 0
        # Default space (in marlin Y units) between frames on the reel
        frame_spacing = 21
        # List of positions (marlin y) where last frames were captured/found
        last_y_list = []

        # Reset Marlin to be zero (homed!!)
        SendMarlinCmd(marlin, "G92 X0 Y0 Z0")

        manual_control=False

        try:
            while True:
                # Check keyboard
                k = cv.waitKey(1) & 0xFF

                if k == 27:    # Esc key to stop/abort
                    break

                if manual_control==True:
                    # Space
                    if k == 32:
                        print("Manual control cancelled")
                        manual_control=False

                    # Manual reel control (for when sproket is not detected)
                    if k == ord('f'):
                        marlin_y += 1
                        MoveFilm(marlin, marlin_y, 500)

                    if k == ord('b'):
                        marlin_y -= 1
                        MoveFilm(marlin, marlin_y, 500)

                # Centre returns the middle of the sproket (if visible)
                # Frame is the picture (already pre-processed)

                # Sometimes OpenCV doesn't detect centre in a particular frame, so try up to 10 times with new
                # camera images before giving up...
                for n in range(0, 10):
                    my_frame, centre = ProcessImage(videoCaptureObject, centre_box, video_width, video_height)
                    frame_counter += 1
                    if centre != None:
                        break

                if manual_control == True:
                    cv.putText(my_frame, "Manual Control Active, keys f/b to align and SPACE to continue", (0, 300), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)

                if centre == None:
                    cv.putText(my_frame, "SPROKET LOST", (0, 200),
                               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv.LINE_AA)

                # Display the raw frame on screen
                cv.putText(my_frame, "{:08d}".format(frame_counter), (0, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)

                cv.imshow('RawVideo', my_frame)

                if centre == None:
                    # We don't have a WHOLE sproket visible on the photo (may be partial ones)
                    # Stop and allow user/manual alignment
                    manual_control=True
                    continue

                if manual_control == True:
                    # Don't process frames in manual alignment mode
                    continue

                if pointInRect(centre, centre_box) == False:
                    # We have a complete sproket visible, but not in the centre of the frame...
                    # Nudge forward until we find the sproket hole centre
                    #print("Advance until sproket in centre frame")
                    marlin_y += 1
                    MoveFilm(marlin, marlin_y, NUDGE_FEED_RATE)
                    continue

                # We have just found our sproket

                print("Found frame {0} at position {1}".format(frame_number, marlin_y))

                # Take a fresh photo now the motion has stopped, ensure the centre is calculated...
                for n in range(0, 10):
                    freeze_frame, centre = ProcessImage(videoCaptureObject, centre_box, video_width, video_height)
                    if centre != None:
                        break

                if centre == None:
                    print("ERROR: Lost frame - Freeze frame photo didn't find sproket!")
                    continue

                # Double check the sproket is still in the correct place...
                if pointInRect(centre, centre_box):
                    try:
                        x = 0
                        # original height is 720 pixels, cut frame of super 8 down to 620 pixels height
                        h = 620
                        y = int(centre[1])-int(h/2)
                        # width 1280 pixels
                        w = freeze_frame.shape[:2][1]
                        #print("Freeze Image", freeze_frame.shape[:2])
                        #print("Crop to x,y,w,h=",x,y,w,h)
                        # Cut out the segment we want to keep - this is now positioned so the sproket
                        # hole is always in the same location
                        crop_img = freeze_frame[y:y+h, x:x+w].copy()

                        #print("crop_img dims", crop_img.shape[:2])

                        # Debug output, mark image with frame number
                        frame_text = "{:08d}".format(frame_number)
                        cv.putText(crop_img, frame_text, (0, 50),cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv.LINE_AA)

                        filename = os.path.join(
                            path, "frame_{:08d}.png".format(frame_number))
                        frame_number += 1

                        # Save frame to disk
                        cv.imwrite(filename, crop_img)

                        # Show it on screen
                        cv.imshow('output', crop_img)

                        # Determine the average gap between captured frames
                        steps = []
                        for n in range(0, len(last_y_list)-1, 2):
                            steps.append(last_y_list[n+1]-last_y_list[n])

                        if len(steps) > 0:
                            total_steps = 0
                            for n in steps:
                                total_steps += n

                            average_spacing = round(total_steps/len(steps), 1)
                            previous_frame_y = last_y_list[len(last_y_list)-1]
                            last_frame_spacing = round(
                                marlin_y-previous_frame_y, 2)

                            print("Average Marlin steps between frames",
                                  average_spacing, ", last frame=", last_frame_spacing)

                            if last_frame_spacing > (average_spacing*1.5):
                                print("Likely dropped frame")
                                # Clear average out after a dropped frame :-(
                                last_y_list = []

                        # Now add on our new reading
                        last_y_list.append(marlin_y)
                        if len(last_y_list) > 20:
                            # Keep list at 20 items, remove first
                            last_y_list.pop(0)

                    except BaseException as CropErr:
                        cv.imshow('output', freeze_frame)
                        print(f"Unexpected {CropErr=}, {type(CropErr)=}")

                    # Now move film forward past the sproket hole so we don't take the same frame twice
                    # do this at a faster speed, to improve captured frames per second
                    marlin_y += frame_spacing
                    MoveFilm(marlin, marlin_y, STANDARD_FEED_RATE)

        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print("Press any key to shut down")
            cv.waitKey()

    # Finished/Quit....
    DisconnectFromMarlin(marlin)
    videoCaptureObject.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
