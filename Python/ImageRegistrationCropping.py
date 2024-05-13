from __future__ import annotations
from typing import TYPE_CHECKING, List
from threading import Thread

import cv2 as cv
import numpy as np
import os
import glob
import shutil
import traceback
import json
import queue

NUM_THREADS = 2

q = queue.Queue(maxsize=10)

min_x=999999
max_x=0
min_y=999999
max_y=0
lower_t=225
previous_frame_top_left_of_sproket_hole=None
previous_frame_bottom_right_of_sproket_hole=None
frame_dims=(0,0, 800, 540)

def ServiceImageWriteQueue(q):
    while True:
        data=q.get(block=True, timeout=None)

        if cv.imwrite(data["filename"], data["image"], [cv.IMWRITE_PNG_COMPRESSION,1])==False:
            raise IOError("Failed to save image")

        q.task_done()

def OutputFolder(path: str) -> str:
    # Create folders for the different EV exposure levels
    
    # Image Output path - create if needed
    path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        os.mkdir(path)

    return path

def ImageFolder(path: str) -> str:  
    # Image Input path - create if needed
    path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return path

def Filelist(path: str, extensions: list) -> list:
    """ Generate a list of files from given path with specified extensions, sorted by filename. """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(path, f"frame_????????.{ext}")))
    files = sorted(files)  # Sort the files after collecting all extensions
    return files

# For Details Reference Link:
# http://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def detectSproket(sproket_image, lower_threshold:int=210):
    # Convert to gray and blur
    matrix = (3, 7)
    sproket_image = cv.GaussianBlur(sproket_image, matrix, 0)
    
    sproket_image = cv.cvtColor(sproket_image, cv.COLOR_BGR2GRAY)

    sproket_image = cv.equalizeHist(sproket_image)
    # Threshold
    _, sproket_image = cv.threshold(sproket_image, lower_threshold, 255, cv.THRESH_BINARY)    

    cv.imshow("sproket_image",cv.resize(sproket_image, (0,0), fx=0.4, fy=0.4))

    # Detect the sproket shape
    contours, _ = cv.findContours(sproket_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #cv.drawContours(image, contours, -1,color=(0,0,255), thickness=cv.FILLED)

    #Abort here if detection found nothing!
    if len(contours)==0:
        #Return a fake reading (hard coded)
        return (455, 646), (455, 646),1,1, 0, 1, len(contours)

    # Sort by area, largest first (hopefully our sproket - we should only have 1 full sprocket in view at any 1 time)
    contour = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[0]      

    #colour = (100, 100, 100)
    #cv.drawContours(sproket_image, [contour], -1,color=colour, thickness=cv.FILLED)

    area = cv.contourArea(contour)
    rect = cv.minAreaRect(contour)
    rotation = rect[2]
    centre = rect[0]
    # Gets center of rotated rectangle
    box = cv.boxPoints(rect)
    # Convert dimensions to ints
    box = np.int0(box)

    #print("area",area)
    #print("rotation",rotation)

    a=min(box[0][0],box[1][0],box[2][0],box[3][0])
    b=min(box[0][1],box[1][1],box[2][1],box[3][1])
    top_left_of_sproket_hole=(a,b)

    a=max(box[0][0],box[1][0],box[2][0],box[3][0])
    b=max(box[0][1],box[1][1],box[2][1],box[3][1])
    bottom_right_of_sproket_hole=(a,b)
    #cv.drawContours(sproket_image, [box], -1,color=(200, 0, 0), thickness=2)

    # Check for vertical stretch
    height_of_sproket_hole=bottom_right_of_sproket_hole[1]-top_left_of_sproket_hole[1]
 
    # Check width
    width_of_sproket_hole=bottom_right_of_sproket_hole[0]-top_left_of_sproket_hole[0]

    #print(top_left_of_sproket_hole, bottom_right_of_sproket_hole)
    #cv.rectangle(sproket_image, top_left_of_sproket_hole, bottom_right_of_sproket_hole, 255, 4)

    return top_left_of_sproket_hole, bottom_right_of_sproket_hole,width_of_sproket_hole,height_of_sproket_hole, rotation, area, len(contours)

def cropOriginalImage(image):
    return image.copy()
    #y1=140
    #y2=y1+2000
    #return image[y1:y2,150:2900].copy()

def scanImageForAverageCalculations(image):
    # Do inital crop of the input image
    # this assumes hardcoded image sizes and will need tweaks depending on input resolution
    image=cropOriginalImage(image)
    h, w =image.shape[:2]

    #Take a vertical strip where the sproket should be (left hand side)
    #Original image is 3556x2381
    y1=int(h*0.2)
    y2=int(h*0.8)
    top_left_of_sproket_hole, bottom_right_of_sproket_hole,width_of_sproket_hole,height_of_sproket_hole, rotation, area, number_of_contours=detectSproket(image[y1:y2,0:int(w*0.20)],lower_threshold=205)

    cv.waitKey(15)

    #Only 1 shape detected, and no rotation
    if number_of_contours<10 and (rotation==0.0 or rotation==90.0 or (rotation>0 and rotation<1)):
        top_left_of_sproket_hole=(top_left_of_sproket_hole[0],y1+top_left_of_sproket_hole[1])
        bottom_right_of_sproket_hole=(bottom_right_of_sproket_hole[0],y1+bottom_right_of_sproket_hole[1])
        cv.rectangle(image, top_left_of_sproket_hole, bottom_right_of_sproket_hole, (0,0,255), 2)
        thumbnail=cv.resize(image, (0,0), fx=0.4, fy=0.4)
        return thumbnail, width_of_sproket_hole,height_of_sproket_hole, area

    return None, None,None,None

def scanImages(files:List, maximum_number_of_samples:int=32):
    # Scan a selection of images looking for "perfect" frames to determine correct
    # size of sproket holes.
    # Asks for human confirmation during the process
    average_sample_count=0
    average_height=0
    average_width=0
    average_area=0

    # Scan first 100 frames/images to determine what "good" looks like
    for filename in files:
        # Quit if we have enough samples
        if average_sample_count>maximum_number_of_samples:
            break

        img = cv.imread(filename,cv.IMREAD_UNCHANGED)
        if img is None:
            print("Error reading",filename)
        else:
            thumbnail, width_of_sproket_hole,height_of_sproket_hole, area=scanImageForAverageCalculations(img)

            if width_of_sproket_hole!=None:
                #Show thumbnail

                cv.putText(thumbnail, "Accept frame? y or n", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 250), 2, cv.LINE_AA)
                cv.putText(thumbnail, "w={0} h={1} area={2}".format(width_of_sproket_hole, height_of_sproket_hole, area), (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 250), 2, cv.LINE_AA)
                cv.putText(thumbnail, "valid samples={0}".format(average_sample_count), (0, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 250), 2, cv.LINE_AA)
                
                cv.imshow("raw",thumbnail)
                thumbnail=None

                k = cv.waitKey(0) & 0xFF

                if k == ord('y'):
                    cv.destroyWindow("raw")
                    average_sample_count+=1
                    average_height+=height_of_sproket_hole
                    average_width+=width_of_sproket_hole
                    average_area+=area

                if k == 27:
                    return_value = False
                    break

                if (average_sample_count>20):
                    break

# samples= 16 w= 352 h= 443 area= 151601
    if (average_sample_count<10):
        raise Exception("Unable to detect suitable sample size")

    # Determine averages
    average_height=int(average_height/average_sample_count)
    average_width=int(average_width/average_sample_count)
    average_area=int(average_area/average_sample_count)

    return average_sample_count,average_width,average_height,average_area

def resize_with_aspect(w, h, delta_w):
    new_w = w + delta_w
    ratio = new_w / w
    new_h = h * ratio
    return new_w, int(new_h), ratio

def processImage(original_image, average_width, average_height, init_frame_dims):
    global min_x,max_x,min_y,max_y, lower_t
    global previous_frame_top_left_of_sproket_hole
    global previous_frame_bottom_right_of_sproket_hole
    global frame_dims

    Detect=True
    manual_adjustment=False
    first_frame=True
    #lower_t=210
    while True:        
        # Do inital crop of the input image
        # this assumes hardcoded image sizes and will need tweaks depending on input resolution
        image=cropOriginalImage(original_image)
        h, w =image.shape[:2]

        if min_x != 999999 or max_x != 0 or min_y != 999999 or max_y != 0:
            first_frame=False

        if Detect:
            #Take a vertical strip where the sproket should be (left hand side)
            top_left_of_sproket_hole, bottom_right_of_sproket_hole, _, _, _, _, _=detectSproket(image[0:h,0:int(w*0.22)], lower_t)

        untouched_image=image.copy()

        # draw actual detected sproket hole in blue
        cv.rectangle(image, top_left_of_sproket_hole, bottom_right_of_sproket_hole, (255,255,100), 3)

        # Draw the box of recorded allowable TOP RIGHT positions (just for fun)
        if max_x>0:
            cv.rectangle(image, (min_x,min_y), (max_x,max_y), (100,255,100), 3)

        #Draw "average" size rectangle in red, based on detected hole
        tl=(bottom_right_of_sproket_hole[0]-average_width,top_left_of_sproket_hole[1])
        br=(tl[0]+average_width,tl[1]+average_height)
        #Top right
        tr=(br[0],tl[1])

        draw_border(image,tl,br,(0,0,255),6,50,40)

        if init_frame_dims != None:
            frame_dims=init_frame_dims
            init_frame_dims=None
        
        frame_tl=(int(tr[0]+frame_dims[0]) ,int(tr[1] + frame_dims[1]))
        frame_br=(int(frame_tl[0]+ frame_dims[2]),int(frame_tl[1]+ frame_dims[3]))

        cv.rectangle(image, frame_tl, frame_br, (0,200,200), 8)
        output_w= frame_br[0]-frame_tl[0]
        output_h= frame_br[1]-frame_tl[1]

        # Highlight top right
        cv.circle(image, (int(tr[0]), int(tr[1])), 8, (0, 255, 0), -1)
        #padding=20

        if frame_tl[1]<0 or frame_tl[0]<0:
            print("frame_tl",frame_tl)
            manual_adjustment=True
        elif tr[0]<min_x or tr[0]>max_x or tr[1]<min_y or tr[1]>max_y:
            print("Outside learned bounding box")
            manual_adjustment=True

        SMALL_STEP=2
        LARGE_STEP=10*SMALL_STEP

        if manual_adjustment==True:
            thumbnail=cv.resize(image, (0,0), fx=0.4, fy=0.4)
            cv.putText(thumbnail, "Cursor keys adjust frame capture(small step), SPACE to confirm", (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv.LINE_AA)
            cv.putText(thumbnail, "Numpad keys adjust frame capture(big step)", (0, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv.LINE_AA)
            cv.putText(thumbnail, "[ and ] adjust threshold, current value={0}".format(lower_t), (0, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv.LINE_AA)
            if first_frame==True:
                cv.putText(thumbnail, "Use a,w,s,d or A,W,S,D to move the crop boundaries and +,- or *,/ to change it's size.", (0, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv.LINE_AA)
                cv.putText(thumbnail, "Use q,e or Q,E to change it's aspect ratio.", (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv.LINE_AA)
            cv.imshow("Adjustment",thumbnail)

            k = cv.waitKeyEx(0)
            # print("key",k)

            # Cursor UP
            if k == 2490368 or k == 63232:
                print("up")
                #Move sproket location up
                # change Y coords
                top_left_of_sproket_hole=(top_left_of_sproket_hole[0],top_left_of_sproket_hole[1]-SMALL_STEP)
                bottom_right_of_sproket_hole=(bottom_right_of_sproket_hole[0],bottom_right_of_sproket_hole[1]-SMALL_STEP)
                Detect=False

            # Cursor Down
            if k == 2621440 or k == 63233:
                print("down")
                #Move sproket location down
                # change Y coords
                top_left_of_sproket_hole=(top_left_of_sproket_hole[0],top_left_of_sproket_hole[1]+SMALL_STEP)
                bottom_right_of_sproket_hole=(bottom_right_of_sproket_hole[0],bottom_right_of_sproket_hole[1]+SMALL_STEP)
                Detect=False

            # Cursor Left
            if k == 2424832 or k == 63234:
                print("left")
                #Move sproket location left
                # change X coords
                top_left_of_sproket_hole=(top_left_of_sproket_hole[0]-SMALL_STEP,top_left_of_sproket_hole[1])
                bottom_right_of_sproket_hole=(bottom_right_of_sproket_hole[0]-SMALL_STEP,bottom_right_of_sproket_hole[1])
                Detect=False

            # Cursor Right
            if k == 2555904 or k == 63235:
                print("right")
                #Move sproket location right
                # change X coords
                top_left_of_sproket_hole=(top_left_of_sproket_hole[0]+SMALL_STEP,top_left_of_sproket_hole[1])
                bottom_right_of_sproket_hole=(bottom_right_of_sproket_hole[0]+SMALL_STEP,bottom_right_of_sproket_hole[1])
                Detect=False

            #  8 (numpad up)
            if k == ord('8'):
                print("upn")
                #Move sproket location up
                # change Y coords
                top_left_of_sproket_hole=(top_left_of_sproket_hole[0],top_left_of_sproket_hole[1]-LARGE_STEP)
                bottom_right_of_sproket_hole=(bottom_right_of_sproket_hole[0],bottom_right_of_sproket_hole[1]-LARGE_STEP)
                Detect=False

            # 2 (numpad down)
            if k == ord('2'):
                print("downn")
                #Move sproket location down
                # change Y coords
                top_left_of_sproket_hole=(top_left_of_sproket_hole[0],top_left_of_sproket_hole[1]+LARGE_STEP)
                bottom_right_of_sproket_hole=(bottom_right_of_sproket_hole[0],bottom_right_of_sproket_hole[1]+LARGE_STEP)
                Detect=False

            # 4 (numpad left)
            if k == ord('4'):
                print("leftn")
                #Move sproket location left
                top_left_of_sproket_hole=(top_left_of_sproket_hole[0]-LARGE_STEP,top_left_of_sproket_hole[1])
                bottom_right_of_sproket_hole=(bottom_right_of_sproket_hole[0]-LARGE_STEP,bottom_right_of_sproket_hole[1])
                Detect=False

            # 6 (numpad right)
            if k == ord('6'):
                print("rightn")
                #Move sproket location right
                top_left_of_sproket_hole=(top_left_of_sproket_hole[0]+LARGE_STEP,top_left_of_sproket_hole[1])
                bottom_right_of_sproket_hole=(bottom_right_of_sproket_hole[0]+LARGE_STEP,bottom_right_of_sproket_hole[1])
                Detect=False

            if k == ord('r'):
                print("Reset")
                #Use previous frames locations
                top_left_of_sproket_hole=previous_frame_top_left_of_sproket_hole
                bottom_right_of_sproket_hole=previous_frame_bottom_right_of_sproket_hole
                Detect=False

            if k == 27:
                raise Exception("Abort!")

            if k == ord('['):
                lower_t-=1
                Detect=True

            if k == ord(']'):
                lower_t+=1
                Detect=True

            if first_frame == True:
                if k == ord('-'):
                    new_w, new_h, ratio = resize_with_aspect(frame_dims[2], frame_dims[3], -SMALL_STEP)
                    frame_dims=(frame_dims[0],frame_dims[1]+(SMALL_STEP/2/ratio),new_w,new_h)
                    Detect=False

                if k == ord('+'):
                    new_w, new_h, ratio = resize_with_aspect(frame_dims[2], frame_dims[3], SMALL_STEP)
                    frame_dims=(frame_dims[0],frame_dims[1]-(SMALL_STEP/2/ratio),new_w,new_h)
                    Detect=False

                if k == ord('/'):
                    new_w, new_h, ratio = resize_with_aspect(frame_dims[2], frame_dims[3], -LARGE_STEP)
                    ratio = new_h / new_w
                    frame_dims=(frame_dims[0],frame_dims[1]+(LARGE_STEP/2/ratio),new_w,new_h)
                    Detect=False

                if k == ord('*'):
                    new_w, new_h, ratio = resize_with_aspect(frame_dims[2], frame_dims[3], LARGE_STEP)
                    print(ratio, frame_dims[1])
                    ratio = new_w / new_h
                    frame_dims=(frame_dims[0],frame_dims[1]-(LARGE_STEP/2/ratio),new_w,new_h)
                    Detect=False

                if k == ord('w'):
                    frame_dims=(frame_dims[0],frame_dims[1]-SMALL_STEP,frame_dims[2],frame_dims[3])
                    Detect=False

                if k == ord('s'):
                    frame_dims=(frame_dims[0],frame_dims[1]+SMALL_STEP,frame_dims[2],frame_dims[3])
                    Detect=False

                if k == ord('a'):
                    frame_dims=(frame_dims[0]-SMALL_STEP,frame_dims[1],frame_dims[2],frame_dims[3])
                    Detect=False
                
                if k == ord('d'):
                    frame_dims=(frame_dims[0]+SMALL_STEP,frame_dims[1],frame_dims[2],frame_dims[3])
                    Detect=False

                if k == ord('W'):
                    frame_dims=(frame_dims[0],frame_dims[1]-LARGE_STEP,frame_dims[2],frame_dims[3])
                    Detect=False

                if k == ord('S'):
                    frame_dims=(frame_dims[0],frame_dims[1]+LARGE_STEP,frame_dims[2],frame_dims[3])
                    Detect=False

                if k == ord('A'):
                    frame_dims=(frame_dims[0]-LARGE_STEP,frame_dims[1],frame_dims[2],frame_dims[3])
                    Detect=False
                
                if k == ord('D'):
                    frame_dims=(frame_dims[0]+LARGE_STEP,frame_dims[1],frame_dims[2],frame_dims[3])
                    Detect=False

                if k == ord('q'):
                    frame_dims=(frame_dims[0],frame_dims[1]+(SMALL_STEP/2),frame_dims[2],frame_dims[3]-SMALL_STEP)
                    Detect=False

                if k == ord('e'):
                    frame_dims=(frame_dims[0],frame_dims[1]-(SMALL_STEP/2),frame_dims[2],frame_dims[3]+SMALL_STEP)
                    Detect=False

                if k == ord('Q'):
                    frame_dims=(frame_dims[0],frame_dims[1]+(LARGE_STEP/2),frame_dims[2],frame_dims[3]-LARGE_STEP)
                    Detect=False
                
                if k == ord('E'):
                    frame_dims=(frame_dims[0],frame_dims[1]-(LARGE_STEP/2),frame_dims[2],frame_dims[3]+LARGE_STEP)
                    Detect=False

                print("frame_dims",frame_dims)

            if k == ord(' '):
                #Accept
                cv.destroyWindow("Adjustment")
                manual_adjustment=False

        if manual_adjustment==False:
            #Black out the sproket hole
            # cv.rectangle(untouched_image,(tr[0]+1,tr[1]-1),(tr[0]-2-average_width,tr[1]+2+average_height),color=(0,0,0),thickness=cv.FILLED)

            if frame_tl[1]<0:
                #Original image is smaller than the crop size/frame size, so pad out
                #Need to pad out the image at the TOP...
                offset_y=abs(frame_tl[1])
                cropped=untouched_image[0:frame_br[1],frame_tl[0]:frame_br[0]].copy()
                h, w =cropped.shape[:2]
                # Full sized image
                output_image = np.zeros((output_h,output_w,3), np.uint8)
                # Place cropped into bottom right corner
                output_image[offset_y:offset_y+h,0:w]=cropped
                return output_image            

            # Update our acceptable min/max ranges
            min_x= min(min_x,tr[0])
            max_x= max(max_x,tr[0])

            min_y= min(min_y,tr[1])
            max_y= max(max_y,tr[1])

            #print(min_x,min_y,max_x,max_y)
            previous_frame_top_left_of_sproket_hole=top_left_of_sproket_hole
            previous_frame_bottom_right_of_sproket_hole=bottom_right_of_sproket_hole

            return untouched_image[frame_tl[1]:frame_br[1],frame_tl[0]:frame_br[0]]

def load_config(config_file):
    """ Load the configuration from a file. """
    try:
        with open(config_file, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_config(config_file, data):
    """ Save the configuration to a file. """
    with open(config_file, 'w') as file:
        json.dump(data, file, indent=4)    

try:
    config_file = 'config.json'
    config = load_config(config_file)
    
    if config is None:
        config = {
            "input_path": "Capture",
            "output_path": "Aligned",
            "extensions": ['png', 'jpg', 'jpeg']
        }

    input_path=ImageFolder(config['input_path'])
    output_path=OutputFolder(config['output_path'])

    print("Input path:", input_path)
    print("Output path:", output_path)

    files=Filelist(input_path, config['extensions'])
    print("Number of files found:", len(files))

    if config.get('average_sample_count') is None:
        print("No config found, redoing analysis.")
        config['average_sample_count'], config['average_width'], config['average_height'], config['average_area'] = scanImages(files[:300])
        save_config(config_file, config)
    else:
        print("Loaded config:", config)

    # Ask user if they want to redo the sprocket analysis
    redo_analysis = input("Do you want to redo the sprocket analysis? (yes/[no]): ").strip().lower()
    if redo_analysis == 'yes' or redo_analysis == 'y':
        config['average_sample_count'], config['average_width'], config['average_height'], config['average_area'] = scanImages(files[:300])
        save_config(config_file, config)
        print("Analysis done and config saved.")
    else:
        print("Proceeding with existing config.")

    average_sample_count = config['average_sample_count']
    average_width = config['average_width']
    average_height = config['average_height']
    average_area = config['average_area']
    print("Using these sprocket values: samples=", average_sample_count, "w=", average_width, "h=", average_height, "area=", average_area)
    
    previous_output_image_filename=None 

    for i in range(NUM_THREADS):
        worker = Thread(target=ServiceImageWriteQueue, args=(q,))
        worker.setDaemon(True)
        worker.start()

    for filename in files:
        new_filename = os.path.join(output_path, os.path.basename(filename))

        #Skip images which already exist
        if os.path.exists(new_filename):
            continue 

        img = cv.imread(filename,cv.IMREAD_UNCHANGED)
        if img is None:
            print("Error opening file",filename,"replacing bad frame")
            #Clone frame to cover up corrupt/missing file
            shutil.copy2(previous_output_image_filename, new_filename)
        else:
            print(filename)

            new_image=processImage(img,  average_width, average_height, config.get('frame_dims'))
            h, w =new_image.shape[:2]


            # Resize image and put into 16:9 frame?
            if config['resize_image'] and config['resize_image'][0] > 0 and config['resize_image'][1] > 0:
                #Output a slightly higher resolution - use post editing to resize
                #this outputs at 16:9 scale

                output_w=config['resize_image'][0]
                output_h=config['resize_image'][1]

                #Scale new_image to keep correct aspect ratio
                scale = output_w/w
                if h*scale > output_h:
                    scale = output_h/h

                scale_w=int(w*scale)
                scale_h=int(h*scale)

                print("Scaled image w=",scale_w,"h=",scale_h, "original w=",w,"h=",h)
                #Horizontal centre frame
                scale_x_offset=int(output_w/2 - scale_w/2)
                scaled_image=cv.resize(new_image, (scale_w,scale_h), interpolation=cv.INTER_AREA)
                
                new_image = np.zeros((output_h,output_w,3), np.uint8)
                new_image[0:scale_h,scale_x_offset:scale_x_offset+scale_w]=scaled_image

            previous_output_image_filename=new_filename

            q.put( {"filename":new_filename, "image":new_image} )

            #Show thumbnail at 50% of original
            thumbnail=cv.resize(new_image, (0,0), fx=0.4, fy=0.4)
            cv.imshow("Final",thumbnail)

            k = cv.waitKey(1) & 0xFF

            if k == 27:
                return_value = False
                break

except BaseException as err:
    print(f"Unexpected {err=}")
    traceback.print_exc()
    print("Press any key to shut down")
    cv.waitKey()

finally:
    print("Waiting for image write queue to empty... length=",q.qsize())
    q.join()
    cv.destroyAllWindows()
