# Super8Scanner.py
#
# (c)2021 Stuart Pittaway
#
# The purpose of this program is to digitize Super8 film reel using an inexpensive USB style camera
# it uses OpenCV to detect the alignment of the images using the film reel sprokets as alignment targets.
# It outputs a PNG image per frame, which are vertically aligned, but frame borders and horizontal alignment
# are not cropped, removed or fixed.  This is the job of a second script to complete this work.
#
# Camera images are captured using YUV mode and images saved as PNG to avoid any compression artifacts during
# the capture and alignment processes
#
# Test on Windows 10 using 1M pixel web camera on an exposed PCB (available on Aliexpress etc.)
#
# Expects to control a MARLIN style stepper driver board
# Y axis is used to drive film feed rollers
# Z axis is used to drive film reel take up spool
# FAN output is used to drive LED light for back light of frames

import picamera
import numpy as np
import cv2 as cv
import glob
import os
import serial
import math
#from serial.serialwin32 import Serial
#import serial.tools.list_ports as port_list
from datetime import datetime, timedelta
import time
import subprocess

camera=None

def pointInRect(point, rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False


def MarlinWaitForReply(MarlinSerialPort: serial.Serial, echoToPrint=True) -> bool:
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


def SendMarlinCmd(MarlinSerialPort: serial.Serial, cmd: str) -> bool:
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


def SendMultipleMarlinCmd(MarlinSerialPort: serial.Serial, cmds: list) -> bool:
    for cmd in cmds:
        SendMarlinCmd(MarlinSerialPort, cmd)
    return True


def PrepareImageForOutput(freeze_frame, frame_number:int, output_video_frame_size, y_offset:int, vertical_offset:int, exposure:float):
    print("Prepare Image", frame_number, y_offset, exposure)

    # Create blank (black) image for output
    output_image = np.zeros(output_video_frame_size, np.uint8)
    output_image_height, output_image_width = output_image.shape[:2]
    #print("output_image",output_image_height, output_image_width)

    #print("crop_img dims", crop_img.shape[:2])
    h, w = freeze_frame.shape[:2]
    # print("freeze_frame",h,w)

    # Draw line across centre of image for debug purposes
    #cv.line(freeze_frame,(0,y_offset),(int(w),y_offset),(100,100,100),2)

    #crop_img = freeze_frame[y:y+h, x:x+w].copy()
    # Put the smaller image inside our output_image, but align it on its centre line

    # Horizontal centre line
    x = int(output_image_width/2) - int(w/2)

    # Vertically centre smaller image inside larger one
    y = int(output_image_height/2 - h/2)

    # Use the top Y value for the sproket, as this doesn't appear to bound around as much as the centre point (affected by height)
    y += int(h/2 - y_offset)
    y -= vertical_offset

    # Offset for vertical sproket alignment based on OpenCV detection
    #y += int(h/2 - centre[1])
    y2 = int(y+h)
    x2 = int(x+w)

    # Vertical centre line
    # print(y, y2, x, x2)
    output_image[y:y2, x:x2] = freeze_frame

    # Debug output, mark image with frame number
    cv.putText(output_image, "{:08d}".format(frame_number), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 2, cv.LINE_AA)

    # Generate a video timecode (H:M:S:FRAMES)
    cv.putText(output_image, timecode(frame_number), (0, 100),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)

    # Generate a video timecode (H:M:S:FRAMES)
    cv.putText(output_image, str(exposure), (0, 200),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)

    return output_image

def TakePicture(video_height,video_width):

    # Capture at the requested resolution in BGR format
    large_image=np.empty((video_height,video_width,3), dtype=np.uint8)
    camera.capture(large_image,'bgr',use_video_port=False)

    image_height, image_width = large_image.shape[:2]
    
    # Now trim out the gate frame (plastic), by cropping the image
    # leave the sproket and the edges of the frame visible
    # keep aspect ratio at 4:3

    # Use RATIO 0.09 rather than exact pixels to cater for different resolutions if needed
    y1=int(image_width*0.09)
    y2=image_height-y1
    x1=int(y1/1.33)
    x2=image_width-x1
    large_image = large_image[y1:y2,x1:x2]
    image_height, image_width = large_image.shape[:2]
    return large_image,image_height, image_width


def ProcessImage(centre_box: list, video_width: int, video_height: int, draw_rects=True, exposure_level=-8.0):

    start_time=time.perf_counter()

    # Contour of detected sproket needs to be this large to be classed as valid (area)
    MIN_AREA_OF_SPROKET = 27500
    MAX_AREA_OF_SPROKET = int(MIN_AREA_OF_SPROKET * 1.1)

    large_image,image_height, image_width=TakePicture(video_height,video_width)
   
    print("Image capture",time.perf_counter() - start_time)

    start_time=time.perf_counter()

    # SubType YUY2
    # Take a picture, in raw YUV mode (avoid MJPEG compression/artifacts)
    #video_height, video_width = image.shape[:2]
    #shape = (int(height * 1.5), int(width))
    #image = image.reshape(shape)
    # Convert YUV2 into RGB for OpenCV to use
    #image = cv.cvtColor(image, cv.COLOR_YUV2BGR)

    #if cap == False:
    #    raise IOError("Failed to capture image")

    # Mirror horizontal - sproket is now on left of image
    large_image = cv.flip(large_image,0)

    #cv.imwrite("large_image.jpg", large_image)

    #Crop larger image down, so we only have the sprokets left
    #y1:y2, x1:x2
    x1 = int(centre_box[0])
    x2 = int(centre_box[0]+centre_box[2])
    frame = large_image[0:image_height,x1:x2]

    # Mask left side of image to find the sprokets, crop out words on film like "KODAK LABS"
    # we are looking for a narrow vertical section of the sprokets, not including any film picture
    # or the curved corners of the sproket holes
    #sproket_mask = np.zeros(frame.shape[:2], dtype="uint8")
    #x1 = int(centre_box[0])
    #x2 = int(centre_box[0]+centre_box[2])
    # top-left corner and bottom-right corner
    #cv.rectangle(sproket_mask, (x1, 0), (x2, image_height), 255, -1)
    #masked = cv.bitwise_and(frame, frame, mask=sproket_mask)
    #cv.imshow("masked",masked)

    # Blur the image and convert to grayscale
    matrix = (5, 9)
    frame_blur = cv.GaussianBlur(frame, matrix, 0)
    imgGry = cv.cvtColor(frame_blur, cv.COLOR_BGR2GRAY)

    #print("Step 1",time.perf_counter() - start_time)
    #cv.imwrite("image_grey.jpg", imgGry)

    # Threshold to only keep the sproket data visible (which is now bright white)
    _, threshold = cv.threshold(imgGry, 150, 255, cv.THRESH_BINARY)
    #cv.imshow('threshold', cv.resize(threshold,(x2,int(image_height/4)) ))

    #print("Step 2",time.perf_counter() - start_time)

    # Get contour of the sproket
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if draw_rects:
        # Draw the target centre box we are looking for (just for debug)
        cv.rectangle(large_image, (centre_box[0], centre_box[1]), (centre_box[0]+centre_box[2], centre_box[1]+centre_box[3]), (0, 255, 0), 8)

    # Sort by area, largest first (hopefully our sproket - we should only have 1 full sprocket in view at any 1 time)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    if len(contours) > 0:
        # Just take the first one...
        contour = contours[0]

        # Find area of detected shapes and filter on the larger ones
        area = cv.contourArea(contour)

        # Sproket must be bigger than this to be okay...
        if area > MIN_AREA_OF_SPROKET and area < MAX_AREA_OF_SPROKET:
            # (center(x, y), (width, height), angleofrotation) = cv.minAreaRect(contour)
            rect = cv.minAreaRect(contour)
            rotation = rect[2]
            centre = rect[0]

            #Add on our offset to the centre (so it now aligns with large_image)
            centre=(centre[0]+centre_box[0],centre[1])

            # Gets center of rotated rectangle
            box = cv.boxPoints(rect)
            # Convert dimensions to ints
            box = np.int0(box)
            colour = (0, 0, 255)

            # Mark centre of sproket with a circle
            if draw_rects:
                cv.circle(large_image, (int(centre[0]), int(centre[1])), 48, (0, 100, 100), -1)

                # Draw the rectangle
                #cv.drawContours(large_image, [box], 0, colour, 8)

            print(time.perf_counter() - start_time)
            return large_image, centre, box
        else:
            print("Area is ",area)
            # pass
    else:
        cv.putText(frame, "No contour", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 2, cv.LINE_AA)

    print(time.perf_counter() - start_time)
    return large_image, None, None

def MoveFilm(marlin: serial.Serial, y: float, feed_rate: int):
    SendMarlinCmd(marlin, "G0 Y{0:.4f} F{1}".format(y, feed_rate))
    # Dwell
    #SendMarlinCmd(marlin,"G4 P100")
    # Wait for move complete
    SendMarlinCmd(marlin, "M400")

def MoveReel(marlin: serial.Serial, z: float, feed_rate: int, wait_for_completion=True):
# Used to rewind the reel/take up slack reel onto spool
    SendMarlinCmd(marlin, "G0 Z{0:.4f} F{1}".format(z, feed_rate))
    if wait_for_completion:
        # Wait for move complete
        SendMarlinCmd(marlin, "M400")

def SetMarlinLight(marlin: serial.Serial, level:int=255):
    #print("Light",level)
    if level>0:
        # M106 Light (fan) On @ PWM level S
        SendMarlinCmd(marlin, "M106 S{0}".format(level))
    else:
        # M107 Light Off
        SendMarlinCmd(marlin, "M107")

def ConnectToMarlin():
    #ports = list(port_list.comports())
    # for p in ports:
    #    print (p)

    # Connect to MARLIN
    marlin = serial.Serial(
        port="/dev/ttyUSB0", baudrate=250000, bytesize=8, timeout=5, stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE
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
    # M18 - Disable steppers (after 15 seconds)
    SendMultipleMarlinCmd(
        marlin, ["M502", "G21", "M211 S0", "G90", "G92 X0 Y0 Z0", "M201 Y0", "M18 S15", "M203 X1000.00 Y1000.00 Z5000.00"])

    SetMarlinLight(marlin,255)

    # M92 - Set Axis Steps-per-unit
    # Just a fake number to keep things uniform, 10 steps
    # 8.888 steps for reel motor, 1 unit is 1 degree = 360 degrees per revolution
    SendMarlinCmd(marlin, "M92 Y10 Z8.888888")

    # Wait for movement to complete
    SendMarlinCmd(marlin, "M400")
    return marlin


def DisconnectFromMarlin(serial_port: serial.Serial):
    # M107 Light Off
    # M84 Steppers Off
    SetMarlinLight(serial_port,0)
    SendMultipleMarlinCmd(serial_port, ["M84"])
    serial_port.close()


def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])




def OutputFolder(exposures:list) -> str:
    # Create folders for the different EV exposure levels    
    for e in exposures:
        path = os.path.join(os.getcwd(), "Capture{0}".format(e))
        if not os.path.exists(path):
            os.makedirs(path)

    # Image Output path - create if needed
    path = os.path.join(os.getcwd(), "Capture")

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def StartupAlignment(marlin: serial.Serial, centre_box, video_width, video_height):
    marlin_y = 0
    reel_z = 0

    return_value = False

    while True:
        #SetExposure(videoCaptureObject)
        my_frame, centre, _ = ProcessImage(centre_box, video_width, video_height, True)

        if centre == None:
            cv.putText(my_frame, "Sproket hole not detected",
                       (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
        else:
            cv.putText(my_frame, "Sproket hole detected, press SPACE to start scanning",
                       (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)

        cv.putText(my_frame, "press f to nudge forward, b for back, j to jump forward quickly,",
                   (10, 200), cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 50), 2, cv.LINE_AA)
        cv.putText(my_frame, "r to rewind spool (1 revolution), ESC to quit",
                   (10, 230), cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 50), 2, cv.LINE_AA)

        image_height, image_width = my_frame.shape[:2]
        cv.imshow('RawVideo', cv.resize(my_frame,(int(image_width/4),int(image_height/4)) ))

        # Check keyboard, wait 0.5 second whilst we do that, then refresh the image capture
        k = cv.waitKey(500) & 0xFF

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

        if k == ord('r'):
            # Rewind tape reel
            reel_z -= 360
            MoveReel(marlin, reel_z, 20000, False)

    return return_value

# Generate timecode based on total number of frames


def timecode(n: int) -> str:
    frames = int(n % 18)
    seconds = timedelta(seconds=int((n - frames) / 18))
    return "{0}.{1}".format(seconds, frames)


def determineStartingFrameNumber(path: str, ext: str) -> int:
    existing_files = sorted(glob.glob(os.path.join(
        path, "frame_????????."+ext)), reverse=True)

    if len(existing_files) > 0:
        return 1+int(os.path.basename(existing_files[0]).split('.')[0][6:])

    return 0

def calculateAngleForSpoolTakeUp(inner_diameter_spool: float, frame_height: float, film_thickness: float, frames_on_spool: int, new_frames_to_spool: int) -> float:
    '''Calculate the angle to wind the take up spool forward based on
    known number of frames already on the spool and the amount of frames we want to add.
     May return more than 1 full revolution of the wheel (for example 650 degrees)'''
    r = inner_diameter_spool/2
    existing_tape_length = frame_height*frames_on_spool
    spool_radius = math.sqrt(existing_tape_length *
                             film_thickness / math.pi + r**2)
    circumfrence = 2*math.pi * spool_radius
    arc_length = new_frames_to_spool * frame_height
    angle = arc_length/circumfrence*360
    # print("spool_radius",spool_radius,"circumfrence",circumfrence,"degrees",angle,"arc_length",arc_length)
    return angle

def main():
    print("OpenCV Version",cv.__version__ )

    # Super8 film dimension (in mm).  The image is vertical on the reel
    # so the reel is 8mm wide and the frame is frame_width inside this.
    FRAME_WIDTH_MM = 5.79
    FRAME_HEIGHT_MM = 4.01
    FILM_THICKNESS_MM = 0.150
    INNER_DIAMETER_OF_TAKE_UP_SPOOL_MM = 32.0

    FRAMES_TO_WAIT_UNTIL_SPOOLING = 6

    # One or several exposures to take images with (for USB camera, only 1 really works)
    CAMERA_EXPOSURE = [-8.0]

    # Constants (sort of)
    NUDGE_FEED_RATE = 1000
    STANDARD_FEED_RATE = 12000

    # Number of PIXELS to remove from the vertical alignment of the output image
    VERTICAL_OUTPUT_OFFSET = 75

    path = OutputFolder(CAMERA_EXPOSURE)
    starting_frame_number = determineStartingFrameNumber(path, "png")
    print("Starting at frame number ", starting_frame_number)

    # Calculate the radius of the tape on the take up spool

    #2MPixel
    #video_width=1920
    #video_height=1088

    #4.9Mpixel
    #video_width=2592
    #video_height=1952

    #8Mpixel, aspect ratio 1.33 = 4:3
    video_width=3296
    video_height=2464

    global camera
    # Open the camera and test capture
    camera=picamera.PiCamera(sensor_mode=3)
    camera.resolution = (video_width,video_height)
    camera.framerate = 1

    #Take a picture to work out what the actual image dimension are (after cropping)
    _,image_height, image_width=TakePicture(video_height,video_width)

    print("Camera configured for resolution ", video_width, "x", video_height, " after crop ",image_width,"x",image_height)
    marlin = ConnectToMarlin()

    # This is the trigger rectangle for the sproket identification
    # must be in the centre of the screen without cropping each frame of Super8

    # Default
    #centre_box = [130, 0, 50, 40]
    # X,Y, W, H
    centre_box = [200, 0, 60, 120]
    # Ensure centre_box is in the centre of the video resolution/image size
    centre_box[1] = int(image_height/2-centre_box[3]/2)

    if StartupAlignment(marlin, centre_box, video_width, video_height) == True:
        # Crude FPS calculation
        time_start = datetime.now()

        # Total number of images stored as a unique frame
        frame_number = starting_frame_number

        frames_already_on_spool = frame_number
        frames_to_add_to_spool = 0

        # Position on film reel (in marlin Y units)
        marlin_y = 0.0
        # Default space (in marlin Y units) between frames on the reel
        frame_spacing = 19
        # List of positions (marlin y) where last frames were captured/found
        last_y_list = []

        # Current Z (take up spool) position
        reel_z=0

        # Reset Marlin to be zero (homed!!)
        SendMarlinCmd(marlin, "G92 X0 Y0 Z0")
        # Disable X and Z steppers, so take up spool rotates freely
        SendMarlinCmd(marlin, "M18 X Z")

        manual_control = False

        # Output video size is larger than the capture size to cope with vertical image stabilization
        # these images will need to be further processed to make valid video files
        output_video_frame_size = (
            int(video_height+256), int(video_width+256), 3)

        try:
            micro_adjustment_steps = 0

            while True:
                if frames_to_add_to_spool> FRAMES_TO_WAIT_UNTIL_SPOOLING+3:
                    # We have processed 12 frames, but only wind 10 onto the spool to leave some slack (3 frames worth)
                    angle=calculateAngleForSpoolTakeUp(INNER_DIAMETER_OF_TAKE_UP_SPOOL_MM,FRAME_HEIGHT_MM, FILM_THICKNESS_MM, frames_already_on_spool, FRAMES_TO_WAIT_UNTIL_SPOOLING)
                    #print("Take up spool angle=",angle)                    
                    reel_z-=angle
                    #Move the stepper spool
                    MoveReel(marlin, reel_z, 8000, False)
                    frames_already_on_spool+=FRAMES_TO_WAIT_UNTIL_SPOOLING
                    frames_to_add_to_spool-=FRAMES_TO_WAIT_UNTIL_SPOOLING

                if micro_adjustment_steps > 25:
                    print("Emergency manual mode as too many small adjustments made")
                    manual_control = True

                # Check keyboard
                k = cv.waitKey(1) & 0xFF

                if k == 27:    # Esc key to stop/abort
                    break

                # Enable manual control (pauses capture)
                if k == ord('m') and manual_control == False:
                    manual_control = True

                if manual_control == True:
                    # Space
                    if k == 32:
                        print("Manual control ended")
                        manual_control = False
                        # FPS counter will be screwed up by manual pause
                        # reset the time and counts here
                        starting_frame_number = frame_number
                        time_start = datetime.now()

                    # Manual reel control (for when sproket is not detected)
                    if k == ord('f'):
                        marlin_y += 1
                        MoveFilm(marlin, marlin_y, 500)

                    if k == ord('b'):
                        marlin_y -= 1
                        MoveFilm(marlin, marlin_y, 500)

                # Centre returns the middle of the sproket hole (if visible)
                # Frame is the picture (already pre-processed)

                # Sometimes OpenCV doesn't detect centre in a particular frame, so try up to 10 times with new
                # camera images before giving up...
                for n in range(0, 10):
                    my_frame, centre, _ = ProcessImage(centre_box, video_width, video_height, True, CAMERA_EXPOSURE[0])
                    if centre != None:
                        break
                    print("Regrab image, no centre")

                if frame_number > 0:
                    fps = (frame_number-starting_frame_number) / \
                        (datetime.now()-time_start).total_seconds()
                    cv.putText(my_frame, "Frames {0}".format(
                        frame_number-starting_frame_number), (0, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv.LINE_AA)
                    cv.putText(my_frame, "Capture FPS {0:.2f}".format(
                        fps), (0, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv.LINE_AA)

                if manual_control == True:
                    cv.putText(my_frame, "Manual Control Active, keys f/b to align and SPACE to continue",
                               (0, 300), cv.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv.LINE_AA)

                if centre == None:
                    cv.putText(my_frame, "SPROKET HOLE LOST", (0, 200),
                               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv.LINE_AA)

                # Display the time on screen, just to prove image is updating
                cv.putText(my_frame, datetime.now().strftime("%X"), (0, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

                image_height, image_width = my_frame.shape[:2]
                cv.imshow('RawVideo', cv.resize(my_frame,(int(image_width/4),int(image_height/4)) ))

                if centre == None:
                    # We don't have a WHOLE sproket hole visible on the photo (may be partial ones)
                    # Stop and allow user/manual alignment
                    manual_control = True
                    continue

                if manual_control == True:
                    # Don't process frames in manual alignment mode
                    continue

                if pointInRect(centre, centre_box) == False:
                    # We have a complete sproket hole visible, but not in the centre of the frame...
                    # Nudge forward until we find the sproket hole centre
                    #print("Advance until sproket hole in centre frame")


                    # As a precaution, limit the total number of small adjustments made
                    # per frame, to avoid going in endless loops and damaging the reel
                    micro_adjustment_steps += 1

                    # We could probably do something clever here and work out a single
                    # jump to move forward/backwards depending on distance between centre line and sproket hole
                    # however with a lop sided rubber band pulley, its all over the place!

                    centre_y=int(centre_box[1]+centre_box[3]/2)

                    # How far off are we?
                    diff_pixels = abs(int(centre_y - centre[1]))

                    print(centre)

                    # sproket hole is below centre line, move reel up
                    if centre[1] > centre_y:
                        print("FORWARD!", marlin_y,
                              "diff pixels=", diff_pixels)
                        marlin_y += 1.5
                    else:
                        # sproket if above centre line, move reel down (need to be careful about reverse feeding film reel into gate)
                        # move slowly/small steps
                        print("REVERSE!", marlin_y,
                              "diff pixels=", diff_pixels)
                        # Fixed step distance for reverse
                        marlin_y -= 0.5

                    MoveFilm(marlin, marlin_y, NUDGE_FEED_RATE)
                    continue


                try:
                    # We have just found our sproket in the centre of the image

                    index=0
                    thumbnail_collection_image = np.zeros((int(video_height*0.25 * len(CAMERA_EXPOSURE)),int(video_width*0.25),3), np.uint8)                    

                    for my_exposure in CAMERA_EXPOSURE:
                        # Take a fresh photo now the motion has stopped, ensure the centre is calculated...
                        for n in range(0, 10):
                            #Delay to let the film/camera settle before a new image
                            #SetExposure(videoCaptureObject,my_exposure)
                            #cv.waitKey(500)
                            freeze_frame, centre, sproket_box = ProcessImage(centre_box, video_width, video_height, False, my_exposure)
                            # Exit loop if we have a valid picture
                            if centre != None:
                                break

                        if centre == None:
                            print("ERROR: Lost frame - Freeze frame photo didn't find sproket! (Exposure=",my_exposure)
                            manual_control = True
                            #cv.imshow("freeze_frame",freeze_frame)
                            #continue
                            raise Exception("Lost frame")

                        # Double check the sproket is still in the correct place...
                        if pointInRect(centre, centre_box) == False:
                            print("Freeze frame centre point outside bounds - trying to align again! (Exposure=",my_exposure)
                            #cv.imshow("freeze_frame",freeze_frame)
                            #continue
                            raise Exception("Freeze frame outside bounds")

                        # Generate thumbnail of the different exposures and make a vertical strip of them
                        thumbnail=cv.resize(freeze_frame, (0,0), fx=0.25, fy=0.25)
                        thumnail_height, thumnail_width = thumbnail.shape[:2]
                        #y:y2, x:x2
                        y1=thumnail_height*index
                        y2=y1+thumnail_height
                        thumbnail_collection_image[y1:y2, 0:thumnail_width]=thumbnail
                        cv.imshow("Exposures",thumbnail_collection_image)
                        index+=1
                        cv.waitKey(1)

                        # The Super8 frame is VERTICALLY aligned in the output image, horizontal alignment
                        # is left as a secondary phase in picture correction (along with colour alignment, scratch removal etc.)

                        # Determine lowest Y value from the sproket rectangle, use that to vertically centre the frame
                        sproket_y = sorted(sproket_box, key=lambda y: y[1], reverse=False)[0]
                        print("Found frame {0} at position {1} with Y of sproket hole {2}, exposure {3}".format(frame_number, marlin_y, sproket_y[1], my_exposure))

                        output_image=PrepareImageForOutput(freeze_frame, frame_number, output_video_frame_size, sproket_y[1], VERTICAL_OUTPUT_OFFSET, my_exposure)
                        filename = os.path.join(path+"{0}".format(my_exposure), "frame_{:08d}.png".format(frame_number))

                        # DEBUG MASK FOR OUTPUT IMAGES
                        #output_mask = np.zeros(output_image.shape[:2], dtype="uint8")
                        #cv.rectangle(output_mask, (327,96), (1230,720), 255, -1)
                        #output_image = cv.bitwise_and(output_image, output_image, mask=output_mask)

                        # Save frame to disk, use lower compression to save CPU time (not image quality png = lossless)
                        if cv.imwrite(filename, output_image, [cv.IMWRITE_PNG_COMPRESSION, 2])==False:
                            raise IOError("Failed to save image")
                        
                        #cv.imshow("output",output_image)
                        #cv.waitKey(50)

                    # Change EXPOSURE
                    #if True==True:
                    #    for my_exposure in CAMERA_EXPOSURE[1:]:
                    #        # Take a fresh photo now the motion has stopped, ensure the centre is calculated...
                    #        for n in range(0, 10):
                    #            #Delay to let the film/camera settle before a new image
                    #            cv.waitKey(50)
                    #            freeze_frame, _, _ = ProcessImage(centre_box, video_width, video_height, False, my_exposure, VERTICAL_OUTPUT_OFFSET)
                    #            if centre != None:
                    #                break
                    #        output_image=PrepareImageForOutput(freeze_frame, frame_number, output_video_frame_size, sproket_y[1], VERTICAL_OUTPUT_OFFSET, my_exposure)
                    #        filename = os.path.join(path+"{0}".format(my_exposure), "frame_{:08d}.png".format(frame_number))                       
                    #        if cv.imwrite(filename, output_image, [cv.IMWRITE_PNG_COMPRESSION, 2])==False:
                    #            raise IOError("Failed to save image")

                    # Move frame number on
                    frame_number += 1
                    # Indicate we want to add a frame to the spool
                    frames_to_add_to_spool+=1

                    # Determine the average gap between captured frames
                    #steps = []
                    #for n in range(0, len(last_y_list)-1, 2):
                    #    steps.append(last_y_list[n+1]-last_y_list[n])
                    #if len(steps) > 0:
                    #    total_steps = 0
                    #    for n in steps:
                    #        total_steps += n
                    #    average_spacing = round(total_steps/len(steps), 1)
                    #    previous_frame_y = last_y_list[len(last_y_list)-1]
                    #    last_frame_spacing = round(
                    #        marlin_y-previous_frame_y, 2)
                    #    print("Average Marlin steps between frames",
                    #          average_spacing, ", last frame=", last_frame_spacing)
                    #    if last_frame_spacing > (average_spacing*1.5):
                    #        print("Likely dropped frame")
                    #        # Clear average out after a dropped frame :-(
                    #        last_y_list = []
                    # Now add on our new reading
                    #last_y_list.append(marlin_y)
                    #if len(last_y_list) > 20:
                    #    # Keep list at 20 items, remove first
                    #    last_y_list.pop(0)

                    # Now move film forward past the sproket hole so we don't take the same frame twice
                    # do this at a faster speed, to improve captured frames per second
                    marlin_y += frame_spacing
                    MoveFilm(marlin, marlin_y, STANDARD_FEED_RATE)
                    micro_adjustment_steps = 0

                except BaseException as err:
                    print(f"Error {err=}, {type(err)=}")

        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print("Press any key to shut down")
            cv.waitKey()

    # Finished/Quit....
    cv.destroyAllWindows()
    camera.close()
    DisconnectFromMarlin(marlin)
   


if __name__ == "__main__":
    main()