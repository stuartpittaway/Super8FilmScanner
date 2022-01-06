import cv2 as cv
import numpy as np
import os
import subprocess

def OutputFolder() -> str:
    # Create folders for the different EV exposure levels
    
    # Image Output path - create if needed
    path = os.path.join(os.getcwd(), "Capture")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return path


def DetectSproketHoleByTemplate(template, img):
    sproket_mask = np.zeros(img.shape[:2], dtype="uint8")
    image_height, image_width = img.shape[:2]
    # top-left corner and bottom-right corner
    cv.rectangle(sproket_mask, (200, 380), (420, 650), 255, -1)
    img = cv.bitwise_and(img, img, mask=sproket_mask)

    matrix = (3, 9)
    imgGry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    template = cv.imread('sproket_hole_template.png',None)
    templateGry = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    w, h = template.shape[:2]

    res = cv.matchTemplate(imgGry,templateGry,cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    #print(min_val, max_val, min_loc, max_loc)

    top_left = max_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)
    return top_left,bottom_right

#CAMERA_EXPOSURE = [-8.0,-4.0,-10.0]
CAMERA_EXPOSURE = [-8.0]

path=OutputFolder()

template = cv.cvtColor(cv.imread('sproket_hole_template.png',None), cv.COLOR_BGR2GRAY)


for frame_number in range(0,21+1):
    print("Frame",frame_number)

    cmd_line=[]
    cmd_line.append("e:\source\enblend-4.2\enfuse.exe")
    cmd_line.append("-l")
    cmd_line.append("8")
    cmd_line.append("-o")
    cmd_line.append(os.path.join(path, "frame_{:08d}.png".format(frame_number)))

    for my_exposure in CAMERA_EXPOSURE:
        filename = os.path.join(path+"{0}".format(my_exposure), "frame_{:08d}.png".format(frame_number))

        cmd_line.append(filename)

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        #Load the source image
        img = cv.imread(filename,None)
        # Roughly detect the sproket hole by template image
        #top_left,bottom_right = DetectSproketHoleByTemplate(template, img)
        #print(top_left,bottom_right)

        top_left=(237,410)
        bottom_right=(389,603)

        # Create mask over sproket hole
        sproket_mask = np.zeros(img.shape[:2], dtype="uint8")
        image_height, image_width = img.shape[:2]
        # Expand the mask rectangle
        PADDING=16
        top_left = (top_left[0]-PADDING,top_left[1]-PADDING)
        bottom_right = (bottom_right[0]+PADDING,bottom_right[1]+PADDING)
        # top-left corner and bottom-right corner
        cv.rectangle(sproket_mask, top_left, bottom_right, 255, -1)
        # Apply the mask to original image, leaving only the sproket hole visible
        sproket_hole = cv.bitwise_and(img, img, mask=sproket_mask)

        # Convert to gray and blur
        matrix = (5, 5)
        sproket_hole = cv.cvtColor(cv.GaussianBlur(sproket_hole, matrix, 0), cv.COLOR_BGR2GRAY)

        sproket_hole = cv.equalizeHist(sproket_hole)

        # Threshold
        _, threshold = cv.threshold(sproket_hole, 155, 255, cv.THRESH_BINARY)

        canny_edges = cv.Canny(threshold, 100, 200)

        contours, _ = cv.findContours(canny_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Sort by area, largest first (hopefully our sproket - we should only have 1 full sprocket in view at any 1 time)
        contour = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[0]

        area = cv.contourArea(contour)
        rect = cv.minAreaRect(contour)
        rotation = rect[2]
        centre = rect[0]
        # Gets center of rotated rectangle
        box = cv.boxPoints(rect)
        # Convert dimensions to ints
        box = np.int0(box)
        colour = (0, 0, 0)
        cv.drawContours(img, [contour], -1,color=colour, thickness=cv.FILLED)
        
        # Check we have the top left and not bottom left
        if box[0][0]<box[1][0]:
            top_left_of_sproket_hole=box[0]
        else:
            top_left_of_sproket_hole=box[1]

        #print(box[0],box[1])

        # Check for vertical stretch
        height_of_sproket_hole=abs(box[0][1]-box[2][1])
        
        if height_of_sproket_hole<165:
            raise Exception("Something wrong!")

        # Crop the output
        x=top_left_of_sproket_hole[0]+120
        y=top_left_of_sproket_hole[1]-250
        h=655
        w=1080-120

        y2=y+h
        x2=x+w
        cropped = img[y:y2, x:x2].copy()        
       
        #if cv.imwrite(filename, cropped, [cv.IMWRITE_PNG_COMPRESSION, 2])==False:
        #    raise IOError("Failed to save image")

        #cv.rectangle(img, top_left, bottom_right, 255, 2)
        #cv.imshow("detect",img)
        cv.imshow("cropped",cropped)
        cv.waitKey(10)

    # Now call ENFUSE
    subprocess.run(cmd_line)


cv.destroyAllWindows()