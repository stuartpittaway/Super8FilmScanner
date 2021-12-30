#Super8Scanner.py
#
#(c)2021 Stuart Pittaway
#
#
import numpy as np
import cv2 as cv
import os

#print( cv.__version__ )

def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False


path=os.path.join(os.getcwd(),"Capture")

if not os.path.exists(path):
    os.makedirs(path)

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

    # Store x and y coordinates as column vectors
    #xcoord = np.array([[459, 450]]).T
    #ycoord = np.array([[467, 466]]).T
    #    points = np.int0(points) 

        # Check to see if any rectangle points are in our lists
    #    if np.any(np.logical_and(xcoords == points[:,0], ycoords == points[:,1])):
    #        continue

    centre_box=[150,350,50,30]
    cv.rectangle(frame,(centre_box[0],centre_box[1]),(centre_box[0]+centre_box[2],centre_box[1]+centre_box[3]),(0,255,0),2)

    reset_box=[150,100,50,70]
    cv.rectangle(frame,(reset_box[0],reset_box[1]),(reset_box[0]+reset_box[2],reset_box[1]+reset_box[3]),(100,100,0),2)

    #Sort by area, largest first (hopefully our sproket - we should only have 1 full sprocket in view at any 1 time)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    if len(contours)>0:

        #for contour in contours:
        contour=contours[0]

        #Find area of detected shapes and filter on the larger ones
        area = cv.contourArea(contour)
        #print(area)

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

    # 20ms delay
    k=cv.waitKey(10) & 0xFF

    if k==27:    # Esc key to stop
        break

videoCaptureObject.release()
cv.destroyAllWindows()
