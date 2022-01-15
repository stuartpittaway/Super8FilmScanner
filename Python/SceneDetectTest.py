from __future__ import annotations
from typing import TYPE_CHECKING, List

import cv2 as cv
import numpy as np
import os
import glob
import shutil
import traceback

# Copied from
# https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv

for frame_number in range(0,200):

    filename = os.path.join("Capture", "frame_{:08d}.png".format(frame_number))
    picture1 = cv.imread(filename,cv.IMREAD_UNCHANGED)
    if picture1 is None:
        raise Exception("Error reading "+filename)

    filename = os.path.join("Capture", "frame_{:08d}.png".format(frame_number+1))
    picture2 = cv.imread(filename,cv.IMREAD_UNCHANGED)
    if picture2 is None:
        raise Exception("Error reading "+filename)
    
    small_picture1=cv.resize(picture1.copy(), (0,0), fx=0.4, fy=0.4)
    small_picture2=cv.resize(picture2.copy(), (0,0), fx=0.4, fy=0.4)

    template=small_picture1[200:400,200:500].copy()
    # Convert pictures to 8 bit grey
    #small_picture1 = cv.cvtColor(small_picture1, cv.COLOR_BGR2GRAY)
    #small_picture2 = cv.cvtColor(small_picture2, cv.COLOR_BGR2GRAY)

    #picture1_norm = picture1/np.sqrt(np.sum(picture1**2))
    #picture2_norm = picture2/np.sqrt(np.sum(picture2**2))
    #a=np.sum(picture1_norm**2)
    #b=np.sum(picture2_norm*picture1_norm)
    #print(a,b, b/a)

    res = cv.matchTemplate(small_picture2,template,cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    print(min_val, max_val, min_loc, max_loc)
    #w, h = template.shape[2]
    w=300
    h=200
    
    #if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    top_left = min_loc
    #else:
    #top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(small_picture2,top_left, bottom_right, 255, 2)


    if min_val>0.09:
        #cv.imshow("picture1",cv.resize(picture1, (0,0), fx=0.1, fy=0.1))
        cv.imshow("template",cv.resize(template, (0,0), fx=1, fy=1))
        cv.imshow("picture2",cv.resize(small_picture2, (0,0), fx=1, fy=1))
        cv.waitKey(10000)

cv.waitKey(0)
cv.destroyAllWindows()