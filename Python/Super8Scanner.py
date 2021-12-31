#Super8Scanner.py
#
#(c)2021 Stuart Pittaway
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

#print( cv.__version__ )

def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

def MarlinWaitForReply(MarlinSerialPort: Serial, echoToPrint = True) -> bool:
    tstart = datetime.now()

    while True:
        # Wait until there is data waiting in the serial buffer
        if MarlinSerialPort.in_waiting > 0:
            # Read data out of the buffer until a CR/NL is found
            serialString = MarlinSerialPort.readline()

            if echoToPrint:
                if serialString.startswith(b"echo:"):
                    # Print the contents of the serial data
                    print("Marlin R:",serialString.decode("Ascii"))

            if serialString==b"ok\n":
                return True

            # Reset delay since last reception
            tstart = datetime.now()

        else:
            #Abort after X seconds of not receiving anything
            duration = datetime.now()-tstart
            if duration.total_seconds()>3:
                return False

def SendMarlinCmd(MarlinSerialPort: Serial, cmd:str) -> bool:
    #print("Sending GCODE",cmd)

    if MarlinSerialPort.isOpen()==False:
        raise Exception("Port closed")

    #Flush input buffer
    MarlinSerialPort.flushInput()
    MarlinSerialPort.flushOutput()
    MarlinSerialPort.read_all()

    MarlinSerialPort.write(cmd.encode('utf-8'))
    MarlinSerialPort.write(b'\n')
    if MarlinWaitForReply(MarlinSerialPort)==False:
        raise Exception("Bad GCODE command or not a valid reply from Marlin")
    
    return True


def SendMultipleMarlinCmd(MarlinSerialPort: Serial, cmds:list) -> bool:
    for cmd in cmds:
        SendMarlinCmd(MarlinSerialPort,cmd)
    return True


def ProcessImage(videoCaptureObject,centre_box):    
    cap, frame = videoCaptureObject.read()

    if cap==False:
        raise Exception("Failed to capture image")

    # Mirror horizontal
    frame = cv.flip(frame, 1)

    image_width=frame.shape[:2][1]
    image_height=frame.shape[:2][0]

    #Mask out any plastic bits on left/right/top/bottom of frame
    main_mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv.rectangle(main_mask, (75, 0), (image_width-75, image_height), 255, -1)
    frame = cv.bitwise_and(frame, frame, mask=main_mask)

    # a mask is the same size as our image, but has only two pixel
    # values, 0 and 255 -- pixels with a value of 0 (background) are
    # ignored in the original image while mask pixels with a value of
    # 255 (foreground) are allowed to be kept
    mask = np.zeros(frame.shape[:2], dtype="uint8")

    # Mask side of image to find the sprokets, crop out words on film like "KODAK LABS"
    # keep 30% of the right hand part of the picture
    x=int(image_width*0.15)
    cv.rectangle(mask, (130, 0), (x, image_height), 255, -1)
    # apply our mask
    masked = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow("masked",masked)

    # Blur the image
    matrix = (17,7)
    frame_blur = cv.GaussianBlur(masked,matrix,0)
    imgGry = cv.cvtColor(frame_blur, cv.COLOR_BGR2GRAY)

    #cv.imwrite("NewFile_name.jpg",image_blur)
    #cv.imshow("Super8FilmScanner",image_blur)
    _, thrash = cv.threshold(imgGry, 200 , 255, cv.THRESH_BINARY)
    cv.imshow("thrash",thrash)

    # find Canny Edges
    canny_edges = cv.Canny(thrash, 30, 200)
    #cv.imshow("canny_edges",canny_edges)

    contours , _ = cv.findContours(canny_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cv.rectangle(frame,(centre_box[0],centre_box[1]),(centre_box[0]+centre_box[2],centre_box[1]+centre_box[3]),(0,255,0),2)

    #Sort by area, largest first (hopefully our sproket - we should only have 1 full sprocket in view at any 1 time)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    if len(contours)>0:
        #for contour in contours:
        contour=contours[0]

        #Find area of detected shapes and filter on the larger ones
        area = cv.contourArea(contour)

        if area>8500:
            #(center(x, y), (width, height), angleofrotation) = cv.minAreaRect(contour)
            rect = cv.minAreaRect(contour)
            rotation = rect[2]
            # Gets center of rotated rectangle
            box = cv.boxPoints(rect)
            box = np.int0(box)
            centre = rect[0]
            colour=(0, 0, 255)

            # Mark centre of sproket
            cv.circle(frame,(int(centre[0]),int(centre[1])), 8, (0,100,100), -1)

            cv.drawContours(frame, [box], 0, colour, 2)
            return frame,centre
        else:
            print("Area is ",area)
    else:
        cv.putText(frame, "No contour", (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    
    return frame,None


def MoveFilm(marlin:Serial,y:float,feed_rate:int):
    SendMarlinCmd(marlin,"G0 Y{0:.4f} F{1}".format(y, feed_rate))
    #Dwell
    #SendMarlinCmd(marlin,"G4 P100")
    #Wait for move complete
    SendMarlinCmd(marlin,"M400")

path=os.path.join(os.getcwd(),"Capture")

if not os.path.exists(path):
    os.makedirs(path)

#ports = list(port_list.comports())
#for p in ports:
#    print (p)

# Connect to MARLIN
marlin = serial.Serial(
    port="COM5", baudrate=250000, bytesize=8, timeout=5, stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE
)

# After initial connection Marlin sends loads of information which we ignore...
MarlinWaitForReply(marlin,False)

# Send setup commands...
#M502 Hardcoded Default Settings Loaded
#G21 - Millimeter Units
#M211 - Software Endstops (disable)
#G90 - Absolute Positioning
#M106 - Fan On (LED LIGHT)
#G92 - Set Position
#M201 - Set Print Max Acceleration (off)
SendMultipleMarlinCmd(marlin,["M502","G21","M211 S0","G90","M106","G92 X0 Y0 Z0","M201 Y0"])

# M92 - Set Axis Steps-per-unit
# Steps per millimeter - 51.5 for 19mm rubber band wheel (lumpy and not circle!)
# Just a fake number!
SendMarlinCmd(marlin,"M92 Y10")

# Wait for movement to complete
SendMarlinCmd(marlin,"M400")

# Open webcamera
videoCaptureObject = cv.VideoCapture(1,cv.CAP_DSHOW)
videoCaptureObject.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
videoCaptureObject.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

cv.namedWindow("RawVideo")
cv.namedWindow("output")

frame_number=0

slow_step=True

marlin_y=0.0
nudge_feed_rate=320
standard_feed_rate=5000

# This is the trigger rectangle for the sproket identification
# must be in the centre of the screen without cropping each frame
# A frame is W1280 and H720
centre_box=[150,342,50,38]

frame_counter=0
frame_spacing=20
last_y_list=[]

try:
    while True:
        # 5ms delay
        k=cv.waitKey(1) & 0xFF

        if k==27:    # Esc key to stop
            break

        # Centre returns the middle of the sproket
        # Frame is the picture (already pre-processed)
        my_frame, centre = ProcessImage(videoCaptureObject,centre_box)
        cv.putText(my_frame, "{:08d}".format(frame_counter), (0,100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255,255), 2, cv.LINE_AA)
        frame_counter+=1

        cv.imshow('RawVideo',my_frame)
        cv.waitKey(5)

        if centre==None:
            # We don't have a WHOLE sproket visible on the photo (may be partial ones)
            # Nudge forward until we find the sproket hole centre
            #print("Advance until we find a sproket...")            
            marlin_y+=1.0
            MoveFilm(marlin,marlin_y,nudge_feed_rate)
            continue

        if pointInRect(centre,centre_box)==False:
            # We have a complete sproket visible, but not in the centre of the frame...
            # Nudge forward until we find the sproket hole centre
            #print("Advance until sproket in centre frame")
            marlin_y+=0.4
            MoveFilm(marlin,marlin_y,nudge_feed_rate)
            continue

        #cv.waitKey(100)
        #cv.imshow('thrash',thrash)
        #cv.imshow('canny_edges',canny_edges)

        #We have just found our sproket

        #It might get stuck oin a loop here...
        print("Found frame {0} at position {1}".format(frame_number, marlin_y))
        while True:
            #Take a fresh photo now the motion has stopped
            freeze_frame,centre = ProcessImage(videoCaptureObject,centre_box)
            if centre==None:
                print("Second photo didn't find sproket!")
                cv.imshow('output',freeze_frame)
            else:
                #print("New centre",centre)
                break

        # Debug allow motion to stop
        #cv.waitKey(5000)

        #Double check the sproket is still in the correct place...
        if pointInRect(centre,centre_box):
            try:
                
                x=0
                #original height is 720 pixels, cut frame of super 8 down to 620 pixels height
                h=620
                y=int(centre[1])-int(h/2)
                #width 1280 pixels
                w=freeze_frame.shape[:2][1]
                #print("Freeze Image", freeze_frame.shape[:2])
                #print("Crop to x,y,w,h=",x,y,w,h)
                # Cut out the segment we want to keep - this is now positioned so the sproket
                # hole is always in the same location
                crop_img = freeze_frame[y:y+h,x:x+w].copy()

                #print("crop_img dims", crop_img.shape[:2])

                # Debug output, mark image with frame number
                frame_text="{:08d}".format(frame_number)
                cv.putText(crop_img, frame_text, (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

                filename=os.path.join( path,"frame_{:08d}.png".format(frame_number) )
                frame_number+=1

                # Save frame to disk
                #print("crop_img dims", crop_img.shape[:2])
                cv.imwrite(filename, crop_img)

                # Show it on screen
                cv.imshow('output',crop_img)

                    
                #Determine the average gap between captured frames
                steps=[]
                for n in range(0,len(last_y_list)-1,2):
                    steps.append(last_y_list[n+1]-last_y_list[n])

                if len(steps)>0:
                    total_steps=0
                    for n in steps:
                        total_steps+=n

                    average_spacing=round(total_steps/len(steps),1)
                    previous_frame_y=last_y_list[len(last_y_list)-1]                   
                    last_frame_spacing=marlin_y-previous_frame_y

                    print("Average Marlin steps between frames", average_spacing,", last frame=",last_frame_spacing)

                    if last_frame_spacing>(average_spacing*1.5):
                        print("Likely dropped frame")
                        # Clear average out after a dropped frame :-(
                        last_y_list=[]

                #Now add on our new reading
                last_y_list.append(marlin_y)
                if len(last_y_list)>10:
                    # Keep list at 10 items, remove first
                    last_y_list.pop(0)

            except BaseException as CropErr:
                cv.imshow('output',freeze_frame)
                print(f"Unexpected {CropErr=}, {type(CropErr)=}")

            # Now move film forward past the sproket hole so we don't take the same frame twice
            # do this at a faster speed, to improve captured frames per second
            marlin_y+=frame_spacing
            MoveFilm(marlin,marlin_y,standard_feed_rate)
       

except BaseException as err:
    print(f"Unexpected {err=}, {type(err)=}")
    print("Press any key to shut down")
    cv.waitKey()

# M107 Light Off
# M84 Steppers Off
SendMultipleMarlinCmd(marlin,["M107","M84"])

videoCaptureObject.release()
cv.destroyAllWindows()
