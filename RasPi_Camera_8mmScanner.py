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

from socket import timeout
from picamera.array import PiRGBArray
from picamera import PiCamera
import queue
from threading import Thread
from fractions import Fraction
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

# Globals (naughty, naughty)
camera = None
shutter_speed = 1000
iso = 50

NUM_THREADS = 3

q = queue.Queue(maxsize=10)

def pointInRect(point, rect):
    if point == None:
        return False
    if rect == None:
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





def GetPreviewImage(large_image):
    preview_image = cv.resize(large_image.copy(), (640, 480))
    image_height, image_width = preview_image.shape[:2]

    # Now trim out the gate frame (plastic), by cropping the image
    # leave the sproket and the edges of the frame visible

    # Use RATIO 0.09 rather than exact pixels to cater for different resolutions if needed
    y1 = 0  # int(image_width*0.02)
    y2 = image_height  # -y1
    x1 = 0  # int(y1/1.33)
    x2 = image_width  # -x1
    preview_image = preview_image[y1:y2, x1:x2].copy()
    image_height, image_width = preview_image.shape[:2]
    return preview_image, image_height, image_width


def ProcessImage(large_image, centre_box: list, draw_rects=True, exposure_level=-8.0, lower_threshold=150):
    # Contour of detected sproket needs to be this large to be classed as valid (area)
    MIN_AREA_OF_SPROKET = 1000
    MAX_AREA_OF_SPROKET = int(MIN_AREA_OF_SPROKET * 14.70)

    preview_image, image_height, image_width = GetPreviewImage(large_image)

    # Crop larger image down, so we only have the sprokets left
    # y1:y2, x1:x2
    x1 = int(centre_box[0])
    x2 = int(centre_box[0]+centre_box[2])
    frame = preview_image[0:image_height, x1:x2]

    # Blur the image and convert to grayscale
    matrix = (5, 9)
    frame_blur = cv.GaussianBlur(frame, matrix, 0)
    imgGry = cv.cvtColor(frame_blur, cv.COLOR_BGR2GRAY)

    # Threshold to only keep the sproket data visible (which is now bright white)
    _, threshold = cv.threshold(imgGry, lower_threshold, 255, cv.THRESH_BINARY)
    #cv.imshow('threshold', threshold)
    # Paste the threshold into the left handside of the preview image to aid visualisation
    preview_image[0:image_height, 0:centre_box[2], 1] = threshold

    # Get contour of the sproket
    contours, _ = cv.findContours(
        threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if draw_rects:
        # Draw the target centre box we are looking for (just for debug, in purple!)
        cv.rectangle(preview_image, (centre_box[0], centre_box[1]), (
            centre_box[0]+centre_box[2], centre_box[1]+centre_box[3]), (128, 0, 128), 2)

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

            # Add on our offset to the centre (so it now aligns with large_image)
            centre = (centre[0]+centre_box[0], centre[1])

            # Gets center of rotated rectangle
            box = cv.boxPoints(rect)
            # Convert dimensions to ints
            box = np.int0(box)
            colour = (200, 0, 200)

            # Mark centre of sproket with a circle
            if draw_rects:
                cv.circle(preview_image, (int(centre[0]), int(
                    centre[1])), 12, (0, 150, 150), -1)

                # Draw the rectangle
                #cv.drawContours(large_image, [box], 0, colour, 8)

            #print(time.perf_counter() - start_time)
            return preview_image, centre, box
        else:
            print("Area is ", area)
            # pass
    else:
        cv.putText(preview_image, "No contour", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    #print(time.perf_counter() - start_time)
    return preview_image, None, None


def MoveFilm(marlin: serial.Serial, y: float, feed_rate: int):
    reversed_y = -y
    SendMarlinCmd(marlin, "G0 Y{0:.4f} F{1}".format(reversed_y, feed_rate))
    # Dwell
    #SendMarlinCmd(marlin,"G4 P100")
    # Wait for move complete
    SendMarlinCmd(marlin, "M400")


def MoveReel(marlin: serial.Serial, z: float, feed_rate: int, wait_for_completion=True):
    # Used to rewind the reel/take up slack reel onto spool
    reversed_z = -z
    SendMarlinCmd(marlin, "G0 Z{0:.4f} F{1}".format(z, feed_rate))
    if wait_for_completion:
        # Wait for move complete
        SendMarlinCmd(marlin, "M400")


def SetMarlinLight(marlin: serial.Serial, level: int = 255):
    # print("Light",level)
    if level > 0:
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

    SetMarlinLight(marlin, 255)

    # M92 - Set Axis Steps-per-unit
    # Just a fake number to keep things uniform, 10 steps
    # 8.888 steps for reel motor, 1 unit is 1 degree = 360 degrees per revolution
    SendMarlinCmd(marlin, "M92 Y10 Z28")

    # Wait for movement to complete
    SendMarlinCmd(marlin, "M400")
    return marlin


def DisconnectFromMarlin(serial_port: serial.Serial):
    # M107 Light Off
    # M84 Steppers Off
    SetMarlinLight(serial_port, 0)
    SendMultipleMarlinCmd(serial_port, ["M84"])
    serial_port.close()


def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def OutputFolder(exposures: list) -> str:
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


lower_threshold = 150


def AutoShutterSpeed(c: PiCamera):
    c.exposure_mode = 'auto'
    c.shutter_speed = shutter_speed
    # Let auto exposure camera do its thing
    time.sleep(2)
    c.exposure_mode = 'off'
    return c.shutter_speed, c.iso


def AutoWB(c: PiCamera, newgain=None):
    if newgain == None:
        c.awb_mode = 'auto'
        # Let AWB do its thing
        time.sleep(2)
        g = c.awb_gains
        # Now lock the white balance
        c.awb_mode = 'off'
        c.awb_gains = g
    else:
        c.awb_mode = 'off'
        c.awb_gains = newgain

    print("awb_mode", c.awb_mode, "awb_gains", c.awb_gains)
    return c.awb_gains

def SetExposure(c: PiCamera, shutter_speed: int = 1000, iso: int = 100):
    print("BEFORE: analog_gain", c.analog_gain, "digital_gain", c.digital_gain)
    1# Fix camera gain and white balance
    if c.iso != iso:
        c.iso = iso
        # Let camera settle
        time.sleep(2)

    #c.exposure_mode = 'auto'
    #time.sleep(0.5)
    c.shutter_speed = shutter_speed
    c.exposure_mode = 'off'
    print("AFTER: iso", c.iso, "exposure_mode", c.exposure_mode, "exposure_speed", c.exposure_speed,
          "shutter_speed", c.shutter_speed)

new_lower_threshold_value=0
def on_startup_threshold_trackbar(val):
    global new_lower_threshold_value
    new_lower_threshold_value=val
    pass

new_shutter_speed_value=0
def on_startup_shutter_speed_trackbar(val):
    global new_shutter_speed_value
    new_shutter_speed_value=val

def StartupAlignment(marlin: serial.Serial, centre_box):
    global lower_threshold, camera
    global new_shutter_speed_value,new_lower_threshold_value
    global shutter_speed, iso



    WINDOW_NAME='Startup Alignment'

    marlin_y = 0
    reel_z = 0

    return_value = False

    configureLowResCamera()

    res = (640, 480)
    rawCapture = PiRGBArray(camera, size=res)

    camera.iso = iso
    # Let camera settle
    time.sleep(2)
    shutter_speed=camera.exposure_speed

    # Set to defaults
    SetExposure(camera, shutter_speed, iso)
    awb_gain=AutoWB(camera)
    #AutoWB(camera, (Fraction(23, 8), Fraction(471, 256)))

    threshold_enable=False

    new_shutter_speed_value=shutter_speed
    new_lower_threshold_value=lower_threshold

   
    cv.namedWindow(WINDOW_NAME)
    trackbar_name = 'Threshold value'
    cv.createTrackbar(trackbar_name, WINDOW_NAME , lower_threshold, 254, on_startup_threshold_trackbar)
    cv.setTrackbarMin(trackbar_name,WINDOW_NAME, 50)

    trackbar_name2 = 'Camera shutter speed'
    cv.createTrackbar(trackbar_name2, WINDOW_NAME , shutter_speed, 100000, on_startup_shutter_speed_trackbar)
    cv.setTrackbarMin(trackbar_name2,WINDOW_NAME, 50)

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        # Capture a small 640x480 image for the preview
        image = frame.array
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        rawCapture.seek(0)

        # Mirror horizontal - sproket is now on left of image
        image = cv.flip(image, 0)

        preview_image, centre, _ = ProcessImage(image, centre_box, True, lower_threshold=lower_threshold)

        # Threshold the entire colour image, this helps find if we have a back light issue
        # and detects hotspots/dark spots
        if threshold_enable:
            _, preview_image = cv.threshold(cv.cvtColor(preview_image, cv.COLOR_BGR2GRAY), lower_threshold, 255, cv.THRESH_BINARY)

        if centre == None:
            cv.putText(preview_image, "Sproket hole not detected",
                       (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv.LINE_AA)
        else:
            cv.putText(preview_image, "Sproket hole detected, press SPACE to start scanning",
                       (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1, cv.LINE_AA)

        # Help text..
        cv.putText(preview_image, "press UP/DOWN to nudge reel, SPACE to cont.",(8, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv.LINE_AA)
        cv.putText(preview_image, "j to jump forward, t toggle threshold style.",(8, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv.LINE_AA)
        cv.putText(preview_image, "Threshold, value={0}".format(lower_threshold),(8, 115), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv.LINE_AA)
        cv.putText(preview_image, "shutter_speed, value={0}, iso={1}".format(shutter_speed, iso), (8, 300), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv.LINE_AA)
        cv.putText(preview_image, "r to rewind spool (1 revolution), ESC to quit", (8, 330), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv.LINE_AA)

        image_height, image_width = preview_image.shape[:2]
        cv.imshow(WINDOW_NAME, preview_image)

        if new_lower_threshold_value!=lower_threshold:
            lower_threshold=new_lower_threshold_value

        if shutter_speed!=new_shutter_speed_value:
            shutter_speed=new_shutter_speed_value
            SetExposure(camera, shutter_speed, iso)

        # Check keyboard, wait whilst we do that, then refresh the image capture
        k = cv.waitKeyEx(30)

        if k == ord(' '):    # SPACE key to continue
            return_value = True
            break

        if k == ord('t'):
            threshold_enable=not(threshold_enable)

        if k == ord('s'):
            shutter_speed, iso = AutoShutterSpeed(camera)

        if k == ord('a'):
            # Set auto white balance and then lock
            AutoWB(camera)

        #Escape
        if k == 27:
            return_value = False
            break

        #Down
        if k == 65362:
            marlin_y += 1
            MoveFilm(marlin, marlin_y, 1000)

        if k == ord('j'):
            marlin_y += 100
            MoveFilm(marlin, marlin_y, 8000)

        #Up
        if k == 65364:
            marlin_y -= 1
            MoveFilm(marlin, marlin_y, 1000)

        if k == ord('r'):
            # Rewind tape reel
            reel_z -= 360
            MoveReel(marlin, reel_z, 20000, False)

    camera.close()
    camera = None
    cv.destroyWindow(WINDOW_NAME)
    return return_value


def determineStartingFrameNumber(path: str, ext: str) -> int:
    existing_files = sorted(glob.glob(os.path.join(
        path, "???????????????????-????????."+ext)), reverse=True)

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

    if camera == None:
        print('Configuring high res camera settings')
        # Close the preview camera object
        # if camera!=None and camera.closed==False:
        #    camera.close()

        # 3840,2496 = 9,584,640pixels
        # 4064,3040 = 12,330,240pixels
        # 3840x2896 = 11,120,640pixels
        # 1920,1440
        # 2880x2166 = 6,266,880pixels
        # 3008x2256 = 6,786,048
        # 3104x2336 = 7,250,944
        res = (3840, 2496)
        #Mode 2
        res = (2048, 1520)
        camera = PiCamera(resolution=res, framerate=30)
        #Mode0 is default, Mode 2 uses binning
        #Mode 2 uses 2028x1520 (half resolution and 2x2binning (softer image))
        camera.sensor_mode=2
        camera.exposure_mode = 'auto'
        camera.awb_mode = 'auto'
        camera.meter_mode = 'backlit'
        #Down the contrast a little (default 0)
        camera.contrast = -10

    return camera.resolution[0], camera.resolution[1]


def configureLowResCamera():
    global camera

    if camera != None and camera.closed == False:
        camera.close()

    res = (640, 480)
    camera = PiCamera(resolution=res, framerate=30)
    #Mode0 is default, Mode 2 uses binning
    camera.sensor_mode=0

    camera.exposure_mode = 'auto'
    camera.awb_mode = 'auto'
    camera.meter_mode = 'backlit'

    #Down the contrast a little (default 0)
    camera.contrast = -10

    return camera.resolution[0], camera.resolution[1]

def ServiceImageWriteQueue(q):

    path = OutputFolder([])

    while True:
        data=q.get(block=True, timeout=None)
        
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-")
        str_current_datetime = str(current_datetime)
        print("Current date & time : ", str_current_datetime)
        
        filename = os.path.join(path+"{0}".format(data["exposure"]), str_current_datetime+"{:08d}".format(data["number"])+".png")
       
        # Save frame to disk.
        # PNG output, with NO compression - which is quicker (less CPU time) on Rasp PI
        # at expense of disk I/O
        # PNG is always lossless
        #start_time = time.perf_counter()
        #if cv.imwrite(filename, data["image"]) == False:
        if cv.imwrite(filename, data["image"], [cv.IMWRITE_PNG_COMPRESSION, 2])==False:
            raise IOError("Failed to save image")
        #print("Save image took {0:.2f} seconds".format(time.perf_counter() - start_time))
        q.task_done()

def main():
    global camera
    print("OpenCV Version", cv.__version__)

    global lower_threshold, shutter_speed, iso

    # Super8 film dimension (in mm).  The image is vertical on the reel
    # so the reel is 8mm wide and the frame is frame_width inside this.
    FRAME_WIDTH_MM = 5.79
    FRAME_HEIGHT_MM = 3.3
    FILM_THICKNESS_MM = 0.150
    INNER_DIAMETER_OF_TAKE_UP_SPOOL_MM = 140.0

    FRAMES_TO_WAIT_UNTIL_SPOOLING = 8

    # One or several exposures to take images with (for USB camera, only 1 really works)
    CAMERA_EXPOSURE = [-1.0]

    # Constants (sort of)
    NUDGE_FEED_RATE = 3000
    STANDARD_FEED_RATE = 12000

    # Number of PIXELS to remove from the vertical alignment of the output image
    #VERTICAL_OUTPUT_OFFSET = 50

    path = OutputFolder(CAMERA_EXPOSURE)
    starting_frame_number = determineStartingFrameNumber(path+"-8.0", "png")
    # starting_frame_number=465bb
    print("Starting at frame number ", starting_frame_number)

    # Calculate the radius of the tape on the take up spool
    camera = None
    highres_width, highres_height = configureHighResCamera()

    # Generate a blank image and pass it through the preview function to determine the cropped size
    preview_image, image_height, image_width = GetPreviewImage(
        np.zeros((highres_height, highres_width, 3), np.uint8))

    print("Camera configured for resolution ", highres_width, "x",
          highres_height, ".  Preview image ", image_width, "x", image_height)

    # This is the trigger rectangle for the sproket identification
    # must be in the centre of the screen without cropping each frame of Super8
    # dimensions are based on the preview window 556x366
    # X,Y, W, H
    centre_box = [40, 65, 40, 40]
    # Ensure centre_box is in the centre of the video resolution/image size
    # we use the PREVIEW sized window for this
    # centre_box[1] = int(image_height/2-centre_box[3]/2)

    marlin = ConnectToMarlin()

    if StartupAlignment(marlin, centre_box) == True:

        # Crude FPS calculation
        time_start = datetime.now()

        # Total number of images stored as a unique frame
        frame_number = starting_frame_number

        frames_already_on_spool = frame_number
        frames_to_add_to_spool = 0

        # Position on film reel (in marlin Y units)
        marlin_y = 0.0
        # Default space (in marlin Y units) between frames on the reel
        FRAME_SPACING = 18
        # List of positions (marlin y) where last frames were captured/found
        last_y_list = []

        # Current Z (take up spool) position
        reel_z = 0

        # Reset Marlin to be zero (homed!!)
        SendMarlinCmd(marlin, "G92 X0 Y0 Z0")
        # Disable X and Z steppers, so take up spool rotates freely
        SendMarlinCmd(marlin, "M18 X Z")

        manual_control = False
    # try:
        micro_adjustment_steps = 0

        for i in range(NUM_THREADS):
            worker = Thread(target=ServiceImageWriteQueue, args=(q,))
            worker.setDaemon(True)
            worker.start()

        # while True:
        highres_width, highres_height = configureHighResCamera()

        SetExposure(camera, shutter_speed, iso)
        #Set AWB after exposure
        AutoWB(camera)

        rawCapture = PiRGBArray(camera, size=(highres_width, highres_height))

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=False):
            #freeze_frame = frame.array
            freeze_frame = frame.array
            rawCapture.truncate(0)
            rawCapture.seek(0)
            # Mirror horizontal - sproket is now on left of image
            freeze_frame = cv.flip(freeze_frame, 0)

            manual_grab = False

            if frames_to_add_to_spool > FRAMES_TO_WAIT_UNTIL_SPOOLING+3:
                # We have processed 12 frames, but only wind 10 onto the spool to leave some slack (3 frames worth)
                angle = calculateAngleForSpoolTakeUp(
                    INNER_DIAMETER_OF_TAKE_UP_SPOOL_MM, FRAME_HEIGHT_MM,
                    FILM_THICKNESS_MM, frames_already_on_spool, FRAMES_TO_WAIT_UNTIL_SPOOLING)
                #print("Take up spool angle=",angle)
                reel_z -= angle
                # Move the stepper spool
                MoveReel(marlin, reel_z, 8000, False)
                frames_already_on_spool += FRAMES_TO_WAIT_UNTIL_SPOOLING
                frames_to_add_to_spool -= FRAMES_TO_WAIT_UNTIL_SPOOLING

            if micro_adjustment_steps > 25:
                print("Emergency manual mode as too many small adjustments made")
                manual_control = True

            # Check keyboard
            if manual_control == True:
                print("Waiting for command key press")
                k = cv.waitKey(10000) & 0xFF
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

                if k == ord(','):
                    shutter_speed -= 50
                    if shutter_speed < 0:
                        shutter_speed = 0
                    SetExposure(camera, shutter_speed, iso)

                if k == ord('.'):
                    shutter_speed += 50
                    if shutter_speed > 180000:
                        shutter_speed = 180000
                    SetExposure(camera, shutter_speed, iso)

                if k == ord('a'):
                    # Set auto white balance and then lock
                    AutoWB(camera)

                # Manual reel control (for when sproket is not detected)
                if k == ord('f'):
                    marlin_y += 1
                    MoveFilm(marlin, marlin_y, 500)

                if k == ord('b'):
                    marlin_y -= 1
                    MoveFilm(marlin, marlin_y, 500)

                if k == ord('['):
                    lower_threshold -= 1

                if k == ord(']'):
                    lower_threshold += 1

                # grab
                if k == ord('g'):
                    # Press g to force capture of a picture, you must ensure the sproket is
                    # manually aligned first
                    #manual_control = False
                    manual_grab = True

            # Centre returns the middle of the sproket hole (if visible)
            # Frame is the picture (already pre-processed)

            # Sometimes OpenCV doesn't detect centre in a particular frame, so try up to 10 times with new
            # camera images before giving up...
            # for n in range(0, 2):
            #    last_exposure=CAMERA_EXPOSURE[0]
            #    freeze_frame,highres_image_height,highres_image_width=TakeHighResPicture()
            #    preview_image, centre, _ = ProcessImage(freeze_frame,centre_box, True, CAMERA_EXPOSURE[0], lower_threshold=lower_threshold)
            #    if centre != None or manual_grab==True or manual_control==True:
            #        break
            #    print("Regrab image, no centre")

            last_exposure = CAMERA_EXPOSURE[0]
            preview_image, centre, _ = ProcessImage(
                freeze_frame, centre_box, True, CAMERA_EXPOSURE[0], lower_threshold=lower_threshold)

            if frame_number > 0:
                fps = (frame_number-starting_frame_number) / \
                    (datetime.now()-time_start).total_seconds()
                cv.putText(preview_image, "Frames {0}, Capture FPS {1:.2f}, fp/h {2:.1f}".format(
                    frame_number-starting_frame_number, fps, fps*3600), (8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv.LINE_AA)
                cv.putText(preview_image, "Threshold {0}".format(
                    lower_threshold), (8, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv.LINE_AA)

            if manual_control == True:
                cv.putText(preview_image, "Manual Control Active, keys f/b to align",
                           (0, 300), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
                cv.putText(preview_image, "[ and ] alter threshold. SPACE to continue",
                           (0, 350), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

            if centre == None and manual_grab == False:
                cv.putText(preview_image, "SPROKET HOLE LOST", (16, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)

            # Display the time on screen, just to prove image is updating
            #cv.putText(preview_image, datetime.now().strftime("%X"), (0, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

            #image_height, image_width = preview_image.shape[:2]
            cv.imshow('RawVideo', preview_image)
            # Let the screen refresh
            cv.waitKey(5)

            if centre == None and manual_grab == False:
                # We don't have a WHOLE sproket hole visible on the photo (may be partial ones)
                # Stop and allow user/manual alignment
                manual_control = True
                continue

            if manual_control == True and manual_grab == False:
                # Don't process frames in manual alignment mode
                continue

            if pointInRect(centre, centre_box) == False and manual_grab == False:
                # We have a complete sproket hole visible, but not in the centre of the frame...
                # Nudge forward until we find the sproket hole centre
                #print("Advance until sproket hole in centre frame")

                # As a precaution, limit the total number of small adjustments made
                # per frame, to avoid going in endless loops and damaging the reel
                micro_adjustment_steps += 1

                # We could probably do something clever here and work out a single
                # jump to move forward/backwards depending on distance between centre line and sproket hole
                # however with a lop sided rubber band pulley, its all over the place!

                centre_y = int(centre_box[1]+centre_box[3]/2)

                # How far off are we?
                diff_pixels = abs(int(centre_y - centre[1]))

                # print(centre)

                # sproket hole is below centre line, move reel up
                if centre[1] > centre_y:
                    print("FORWARD!", marlin_y, "diff pixels=", diff_pixels)
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

                    #if last_exposure == my_exposure:
                    #    highres_image_height, highres_image_width = freeze_frame.shape[:2]
                    #else:
                        # Take a fresh image
                    #    freeze_frame, highres_image_height, highres_image_width = TakeHighResPicture()

                    highres_image_height, highres_image_width = freeze_frame.shape[:2]

                    # Generate thumbnail of the picture and show it
                    thumbnail = cv.resize(freeze_frame, (0, 0), fx=0.50, fy=0.50)
                    thumnail_height, thumnail_width = thumbnail.shape[:2]
                    cv.imshow("Exposure", thumbnail)

                    # Save the image to the queue
                    q.put( {"number":frame_number,"exposure":my_exposure, "image":freeze_frame} )
                    print("Image put onto queue, q length=",q.qsize())

                # Move frame number on
                frame_number += 1
                # Indicate we want to add a frame to the spool
                frames_to_add_to_spool += 1

                # Now move film forward past the sproket hole so we don't take the same frame twice
                # do this at a faster speed, to improve captured frames per second
                marlin_y += FRAME_SPACING
                MoveFilm(marlin, marlin_y, STANDARD_FEED_RATE)
                micro_adjustment_steps = 0

            except BaseException as err:
                print(f"High Res Capture Loop Error {err=}, {type(err)=}")

    # except BaseException as err:
    #    print(f"Unexpected {err=}, {type(err)=}")
    #    print("Press any key to shut down")
    #    cv.waitKey()

    # Finished/Quit....
    print("Waiting for image write queue to empty... length=",q.qsize())
    q.join()
    print("Destroy windows")
    cv.destroyAllWindows()
    print("Disconnect Marlin")
    DisconnectFromMarlin(marlin)

    if camera != None and camera.closed == False:
        camera.close()

if __name__ == "__main__":
    main()
