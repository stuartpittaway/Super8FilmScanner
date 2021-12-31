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

        else:
            #Abort after X seconds of not receiving anything
            duration = datetime.now()-tstart
            if duration.total_seconds()>2:
                return False


def SendMarlinCmd(MarlinSerialPort: Serial, cmd:str) -> bool:
    print("Sending GCODE",cmd)

    if MarlinSerialPort.isOpen()==False:
        raise Exception("Port closed")

    MarlinSerialPort.write(cmd)
    MarlinSerialPort.write(b'\n')
    if MarlinWaitForReply(MarlinSerialPort)==False:
        raise Exception("Bad GCODE command or not a valid reply from Marlin")
    
    return True


def SendMultipleMarlinCmd(MarlinSerialPort: Serial, cmds:list) -> bool:
    for cmd in cmds:
        SendMarlinCmd(MarlinSerialPort,cmd)

    return True


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
#G91 - Relative Positioning
#M106 - Fan On (LED LIGHT)
#G92 - Set Position
SendMultipleMarlinCmd(marlin,[b"M502",b"G21",b"M211 S0",b"G91",b"M106",b"G92 X0 Y0 Z0"])

# M92 - Set Axis Steps-per-unit
# Steps per millimeter - 51.5 for 19mm rubber band wheel (lumpy and not circle!)
SendMarlinCmd(marlin,b"M92 Y51.5")

# Wait for movement to complete
SendMarlinCmd(marlin,b"M400")



# Open webcamera
videoCaptureObject = cv.VideoCapture(1,cv.CAP_DSHOW)
videoCaptureObject.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
videoCaptureObject.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

cv.namedWindow("RawVideo")
cv.namedWindow("output")

delay_for_next_frame=False
frame_number=0

while True:
    cap, frame = videoCaptureObject.read()

    if cap==False:
        raise Exception("Failed to capture image")

    # Mirror horizontal
    frame = cv.flip(frame, 1)

    image_size=frame.shape[:2]

    #Mask out any plastic bits on left/right/top/bottom of frame
    main_mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv.rectangle(main_mask, (75, 0), (image_size[1]-75, image_size[0]), 255, -1)
    frame = cv.bitwise_and(frame, frame, mask=main_mask)

    # a mask is the same size as our image, but has only two pixel
    # values, 0 and 255 -- pixels with a value of 0 (background) are
    # ignored in the original image while mask pixels with a value of
    # 255 (foreground) are allowed to be kept
    mask = np.zeros(frame.shape[:2], dtype="uint8")

    # Mask side of image to find the sprokets, crop out words on film like "KODAK LABS"
    # keep 30% of the right hand part of the picture
    x=int(image_size[1]*0.18)
    cv.rectangle(mask, (130, 0), (x, image_size[0]), 255, -1)
    # apply our mask
    masked = cv.bitwise_and(frame, frame, mask=mask)

    # Blur the image
    matrix = (17,7)
    frame_blur = cv.GaussianBlur(masked,matrix,0)
    imgGry = cv.cvtColor(frame_blur, cv.COLOR_BGR2GRAY)
    frame_blur=None

    #cv.imwrite("NewFile_name.jpg",image_blur)
    #cv.imshow("Super8FilmScanner",image_blur)
    _, thrash = cv.threshold(imgGry, 235 , 255, cv.THRESH_BINARY)
    imgGry=None

    #  find Canny Edges
    canny_edges = cv.Canny(thrash, 30, 200)
    #cv.imshow("canny_edges",canny_edges)

    contours , _ = cv.findContours(canny_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE )

    centre_box=[150,350,50,30]
    cv.rectangle(frame,(centre_box[0],centre_box[1]),(centre_box[0]+centre_box[2],centre_box[1]+centre_box[3]),(0,255,0),2)

    reset_box=[150,250,50,70]
    cv.rectangle(frame,(reset_box[0],reset_box[1]),(reset_box[0]+reset_box[2],reset_box[1]+reset_box[3]),(100,100,0),2)

    #Sort by area, largest first (hopefully our sproket - we should only have 1 full sprocket in view at any 1 time)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    if len(contours)>0:

        #for contour in contours:
        contour=contours[0]

        #Find area of detected shapes and filter on the larger ones
        area = cv.contourArea(contour)

        if area>10000:

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

            # Is the box in the middle of the window/image?
            # make sure its a tight box centre point
            if pointInRect(centre,centre_box) and delay_for_next_frame==False:
            #if centre[0]>170 and centre[0]<180 and centre[1]>350 and centre[1]<360:
                #x=int(centre[0])
                x=0
                #720 pixel image?
                y=int(centre[1])-320
                #1280 pixels
                w=image_size[1]
                h=620
                crop_img = frame[y:y+h,x:x+w].copy()

                # Debug output, mark image with frame number
                frame_text="{:08d}".format(frame_number)
                cv.putText(crop_img, frame_text, (0,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

                filename=os.path.join( path,"frame_{:08d}.png".format(frame_number) )
                frame_number+=1

                cv.imwrite(filename, crop_img) 

                cv.imshow('output',crop_img)

                #Green
                colour=(0, 255, 0)
                delay_for_next_frame=True

            elif pointInRect(centre,reset_box) and delay_for_next_frame==True:
                #Green
                colour=(255, 0, 0)
                delay_for_next_frame=False

            #x, y , w, h = cv.boundingRect(box)
            #print(centre,rotation,area,box)           
            #approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True), True)
            #cv.drawContours(frame, [approx], 0, (0, 255, 0), 2)          
                
            cv.drawContours(frame, [box], 0, colour, 2)
    else:
        print("No contour")

    #cv.imshow('thrash',thrash)
    #cv.imshow('canny_edges',canny_edges)
    cv.imshow('RawVideo',frame)

    # 2ms delay
    k=cv.waitKey(2) & 0xFF

    if k==27:    # Esc key to stop
        break

videoCaptureObject.release()
cv.destroyAllWindows()

# M107 Light Off
# M84 Steppers Off
SendMultipleMarlinCmd(marlin,[b"M107",b"M84"])
