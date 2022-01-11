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
    if point==None:
        return False
    if rect==None:
        return False
        
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



def TakeHighResPicture():
    global camera

    start_time=time.perf_counter()

    video_width,video_height=configureHighResCamera()

    # Capture at the requested resolution in BGR format
    large_image=np.empty((video_height,video_width,3), dtype=np.uint8)
    camera.capture(large_image,'bgr',use_video_port=False)

    # Close it after the image
    camera.close()

    print("High res image capture {:.2f}".format(time.perf_counter() - start_time))
    image_height, image_width = large_image.shape[:2]  

    # Now trim out the gate frame (plastic), by cropping the image
    # leave the sproket and the edges of the frame visible
    # keep aspect ratio at 4:3

    # Use RATIO 0.09 rather than exact pixels to cater for different resolutions if needed
    #y1=int(image_width*0.06)
    #y2=image_height-y1
    #x1=int(y1/1.33)
    #x2=image_width-x1
    #large_image = large_image[y1:y2,x1:x2]
    #image_height, image_width = large_image.shape[:2]

    return large_image,image_height, image_width

def TakePreviewPicture(video_width:int,video_height:int):
    global camera

    if camera==None or camera.closed:
        video_width,video_height=configurePreviewCamera()

    # Capture at the requested resolution in BGR format
    large_image=np.empty((video_height,video_width,3), dtype=np.uint8)
    camera.capture(large_image,'bgr',use_video_port=True)

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


def ProcessImage(centre_box: list, video_width: int, video_height: int, draw_rects=True, exposure_level=-8.0, lower_threshold=200):

    #start_time=time.perf_counter()

    # Contour of detected sproket needs to be this large to be classed as valid (area)
    MIN_AREA_OF_SPROKET = 2650
    MAX_AREA_OF_SPROKET = int(MIN_AREA_OF_SPROKET * 1.25)

    large_image,image_height, image_width=TakePreviewPicture(video_width,video_height)
   
    #print("Image capture",time.perf_counter() - start_time)

    #start_time=time.perf_counter()

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
    _, threshold = cv.threshold(imgGry, lower_threshold, 255, cv.THRESH_BINARY)
    cv.imshow('threshold', threshold)

    #print("Step 2",time.perf_counter() - start_time)

    # Get contour of the sproket
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if draw_rects:
        # Draw the target centre box we are looking for (just for debug, in purple!)
        cv.rectangle(large_image, (centre_box[0], centre_box[1]), (centre_box[0]+centre_box[2], centre_box[1]+centre_box[3]), (128, 0, 128), 2)

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
                cv.circle(large_image, (int(centre[0]), int(centre[1])), 12, (0, 150, 150), -1)

                # Draw the rectangle
                #cv.drawContours(large_image, [box], 0, colour, 8)

            #print(time.perf_counter() - start_time)
            return large_image, centre, box
        else:
            print("Area is ",area)
            # pass
    else:
        cv.putText(frame, "No contour", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)

    #print(time.perf_counter() - start_time)
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

lower_threshold=200

def StartupAlignment(marlin: serial.Serial,centre_box, video_width, video_height):
    global lower_threshold
    marlin_y = 0
    reel_z = 0

    return_value = False

    while True:
        #SetExposure(videoCaptureObject)
        my_frame, centre, _ = ProcessImage(centre_box, video_width, video_height, True, lower_threshold=lower_threshold)

        if centre == None:
            cv.putText(my_frame, "Sproket hole not detected",
                       (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv.LINE_AA)
        else:
            cv.putText(my_frame, "Sproket hole detected, press SPACE to start scanning",
                       (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1, cv.LINE_AA)

        #Help text..
        cv.putText(my_frame, "press f to nudge forward,",
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv.LINE_AA)
        cv.putText(my_frame, "b for back, j to jump forward quickly,",
                   (10, 85), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv.LINE_AA)
        cv.putText(my_frame, "[ and ] alter threshold, value={0}".format(lower_threshold),
                   (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv.LINE_AA)
        cv.putText(my_frame, "r to rewind spool (1 revolution), ESC to quit",
                   (10, 260), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv.LINE_AA)

        image_height, image_width = my_frame.shape[:2]
        cv.imshow('RawVideo',my_frame)

        # Check keyboard, wait 0.5 second whilst we do that, then refresh the image capture
        k = cv.waitKey(500) & 0xFF

        if k == ord(' '):    # SPACE key to continue
            return_value = True
            break

        if k == ord('['):
            lower_threshold-=1

        if k == ord(']'):
            lower_threshold+=1

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

def configureHighResCamera():
    global camera

    # Close the preview camera object
    if camera!=None and camera.closed==False:
        camera.close()

    #10Mpixel
    camera=picamera.PiCamera(resolution=(3840,2496),framerate = 30)
    camera.exposure_mode = 'auto'
    camera.awb_mode='auto'
    camera.meter_mode='backlit'
    return camera.resolution[0],camera.resolution[1]


def configurePreviewCamera():
    global camera

    # Close the preview camera object
    if camera!=None and camera.closed==False:
        camera.close()

    camera=picamera.PiCamera(resolution = (640,480),framerate=30)
    #camera.shutter_speed = 10000
    camera.exposure_mode = 'auto'
    #camera.iso = 1600
    camera.awb_mode='auto'
    camera.meter_mode='backlit'
    return camera.resolution[0],camera.resolution[1]


def main():
    print("OpenCV Version",cv.__version__ )

    global lower_threshold

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
    #VERTICAL_OUTPUT_OFFSET = 50

    path = OutputFolder(CAMERA_EXPOSURE)
    starting_frame_number = determineStartingFrameNumber(path+"-8.0", "bmp")
    #starting_frame_number=465bb
    print("Starting at frame number ", starting_frame_number)

    # Calculate the radius of the tape on the take up spool

    #2MPixel
    #video_width=1920
    #video_height=1088

    #4.9Mpixel
    #video_width=2592
    #video_height=1952


    #video_width=640
    #video_height=480

    video_width,video_height=configurePreviewCamera()
    # Open the camera and test capture
    #camera=picamera.PiCamera()
    #camera.exposure_mode = 'auto'    
    #camera.awb_mode='auto'
    #camera.resolution = (video_width,video_height)
    #camera.framerate = 30

    #Take a picture to work out what the actual image dimension are (after cropping)
    _,image_height, image_width=TakePreviewPicture(video_width,video_height)

    print("Camera configured for resolution ", video_width, "x", video_height, " after crop ",image_width,"x",image_height)
    marlin = ConnectToMarlin()

    # This is the trigger rectangle for the sproket identification
    # must be in the centre of the screen without cropping each frame of Super8

    # Default
    #centre_box = [130, 0, 50, 40]
    # X,Y, W, H
    centre_box = [25, 0, 32, 40]
    # Ensure centre_box is in the centre of the video resolution/image size
    centre_box[1] = int(image_height/2-centre_box[3]/2)

    if StartupAlignment(marlin,centre_box, video_width, video_height) == True:
        # Crude FPS calculation
        time_start = datetime.now()

        # Total number of images stored as a unique frame
        frame_number = starting_frame_number

        frames_already_on_spool = frame_number
        frames_to_add_to_spool = 0

        # Position on film reel (in marlin Y units)
        marlin_y = 0.0
        # Default space (in marlin Y units) between frames on the reel
        frame_spacing = 20
        # List of positions (marlin y) where last frames were captured/found
        last_y_list = []

        # Current Z (take up spool) position
        reel_z=0

        # Reset Marlin to be zero (homed!!)
        SendMarlinCmd(marlin, "G92 X0 Y0 Z0")
        # Disable X and Z steppers, so take up spool rotates freely
        SendMarlinCmd(marlin, "M18 X Z")

        manual_control = False
    #try:
        micro_adjustment_steps = 0

        while True:
            manual_grab=False

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
            if manual_control == True:
                k = cv.waitKey(250) & 0xFF
            else:
                k = cv.waitKey(10) & 0xFF

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

                if k == ord('['):
                    lower_threshold-=1
                    
                if k == ord(']'):
                    lower_threshold+=1

                # grab
                if k == ord('g'):
                    # Press g to force capture of a picture, you must ensure the sproket is 
                    # manually aligned first
                    manual_control=False
                    manual_grab=True

            # Centre returns the middle of the sproket hole (if visible)
            # Frame is the picture (already pre-processed)

            # Sometimes OpenCV doesn't detect centre in a particular frame, so try up to 10 times with new
            # camera images before giving up...
            for n in range(0, 5):
                my_frame, centre, _ = ProcessImage(centre_box, video_width, video_height, True, CAMERA_EXPOSURE[0], lower_threshold=lower_threshold)
                if centre != None or manual_grab==True or manual_control==True:
                    break
                print("Regrab image, no centre")

            if frame_number > 0:
                fps = (frame_number-starting_frame_number) / \
                    (datetime.now()-time_start).total_seconds()
                cv.putText(my_frame, "Frames {0}, Capture FPS {1:.2f}, fp/h {2:.1f}".format(
                    frame_number-starting_frame_number, fps, fps*3600), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv.LINE_AA)

            if manual_control == True:
                cv.putText(my_frame, "Manual Control Active, keys f/b to align and SPACE to continue",
                            (0, 300), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

            if centre == None and manual_grab==False:
                cv.putText(my_frame, "SPROKET HOLE LOST", (16, 100),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)

            # Display the time on screen, just to prove image is updating
            #cv.putText(my_frame, datetime.now().strftime("%X"), (0, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

            image_height, image_width = my_frame.shape[:2]
            cv.imshow('RawVideo', my_frame)
            # Let the screen refresh
            cv.waitKey(3)

            if centre == None and manual_grab==False:
                # We don't have a WHOLE sproket hole visible on the photo (may be partial ones)
                # Stop and allow user/manual alignment
                manual_control = True
                continue

            if manual_control == True and manual_grab==False:
                # Don't process frames in manual alignment mode
                continue

            if pointInRect(centre, centre_box) == False and manual_grab==False:
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

                #print(centre)

                # sproket hole is below centre line, move reel up
                if centre[1] > centre_y:
                    print("FORWARD!", marlin_y,"diff pixels=", diff_pixels)
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
                if manual_grab:
                    print("Manual Grab!")

                # We have just found our sproket in the centre of the image
                for my_exposure in CAMERA_EXPOSURE:
                    # Take a fresh photo now the motion has stopped, ensure the centre is calculated...

                    freeze_frame,highres_image_height,highres_image_width=TakeHighResPicture()

                    #camera image is 3840,2496 = 9.5Megapixels (way overkill for Super8!)
                    #but we need to crop down

                    #crop large image to remove plastic gate edges
                    #use ratios in case the camera resolution changes
                    y1=int(0.003 * highres_image_height)
                    y2=int(0.957 * highres_image_height)
                    x1=int(0.036 * highres_image_width)
                    x2=int(0.962 * highres_image_width)
                    freeze_frame=freeze_frame[y1:y2,x1:x2].copy()
                    highres_image_height,highres_image_width = freeze_frame.shape[:2]

                    #Output image is now 3556x2381 = 8.5Megapixel (still way overkill for Super8!)

                    # Reduce file size a little
                    #freeze_frame = cv.resize(freeze_frame,(0,0), fx=0.8, fy=0.8, interpolation = cv.INTER_AREA)

                    # Mirror - sproket is now on left of image
                    freeze_frame = cv.flip(freeze_frame,0)

                    #for n in range(0, 10):
                    #    #Delay to let the film/camera settle before a new image
                    #    #SetExposure(videoCaptureObject,my_exposure)
                    #    freeze_frame, centre, sproket_box = ProcessImage(centre_box, video_width, video_height, False, my_exposure)
                    #    # Exit loop if we have a valid picture
                    #    if centre != None:
                    #        break

                    #if centre == None:
                    #    print("ERROR: Lost frame - Freeze frame photo didn't find sproket! (Exposure=",my_exposure)
                    #    manual_control = True
                    #    raise Exception("Lost frame")

                    # Double check the sproket is still in the correct place...
                    #if pointInRect(centre, centre_box) == False:
                    #    print("Freeze frame centre point outside bounds - trying to align again! (Exposure=",my_exposure)
                    #    #cv.imshow("freeze_frame",freeze_frame)
                    #    #continue
                    #    raise Exception("Freeze frame outside bounds")

                    # Generate thumbnail of the picture and show it
                    thumbnail=cv.resize(freeze_frame, (0,0), fx=0.30, fy=0.30)
                    thumnail_height, thumnail_width = thumbnail.shape[:2]
                    cv.imshow("Exposure",thumbnail)

                    # The Super8 frame is VERTICALLY aligned in the output image, horizontal alignment
                    # is left as a secondary phase in picture correction (along with colour alignment, scratch removal etc.)
                    # Determine lowest Y value from the sproket rectangle, use that to vertically centre the frame
                    #sproket_y = sorted(sproket_box, key=lambda y: y[1], reverse=False)[0]
                    #print("Found frame {0} at position {1} with Y of sproket hole {2}, exposure {3}".format(frame_number, marlin_y, sproket_y[1], my_exposure))
                    
                    #vertical_scale = highres_image_height/image_height
                    # Output video size is larger than the capture size to cope with vertical image stabilization
                    # these images will need to be further processed to make valid video files
                    # The height allow the image to bounce up and down
                    #output_video_frame_size = (int(highres_image_height+(centre_box[3]*vertical_scale)), int(highres_image_width), 3)
                    #output_image=PrepareImageForOutput(freeze_frame, frame_number, output_video_frame_size, vertical_scale*centre[1])
                    
                    filename = os.path.join(path+"{0}".format(my_exposure), "frame_{:08d}.bmp".format(frame_number))

                    # DEBUG MASK FOR OUTPUT IMAGES
                    #output_mask = np.zeros(output_image.shape[:2], dtype="uint8")
                    #cv.rectangle(output_mask, (327,96), (1230,720), 255, -1)
                    #freeze_frame = cv.bitwise_and(freeze_frame, freeze_frame, mask=output_mask)

                    # Save frame to disk.
                    start_time=time.perf_counter()
                    
                    # PNG output, with NO compression - which is quicker (less CPU time) on Rasp PI
                    # at expense of disk I/O
                    # PNG is always lossless
                    if cv.imwrite(filename, freeze_frame)==False:
                    #if cv.imwrite(filename, freeze_frame, [cv.IMWRITE_PNG_COMPRESSION, 0])==False:
                    #if cv.imwrite(filename, freeze_frame, [cv.IMWRITE_JPEG_QUALITY, 100])==False:
                        raise IOError("Failed to save image")
                    print("Save image took {:.2f} seconds".format(time.perf_counter() - start_time))
                    
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
                print(f"High Res Capture Loop Error {err=}, {type(err)=}")

    #except BaseException as err:
    #    print(f"Unexpected {err=}, {type(err)=}")
    #    print("Press any key to shut down")
    #    cv.waitKey()

    # Finished/Quit....
    cv.destroyAllWindows()
    if camera.closed==False:
        camera.close()
    DisconnectFromMarlin(marlin)



if __name__ == "__main__":
    main()
