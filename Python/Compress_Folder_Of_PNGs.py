import numpy as np
import cv2 as cv
import time

import glob
import os

def Filelist(path: str, ext: str) -> int:
    return sorted(glob.glob(os.path.join(path, "frame_????????."+ext)), reverse=False)

path = os.path.join(os.getcwd(), "Capture")
if not os.path.exists(path):
    raise FileNotFoundError("Missing folder")

for filename in Filelist(path,"png"):
    img = cv.imread(filename,cv.IMREAD_UNCHANGED)
    if img is None:
        print("Error opening file",filename)
    else:
        if os.path.getsize(path)>7500000:
            start_time=time.perf_counter()
            # PNG output, with NO compression - which is quicker (less CPU time) on Rasp PI
            # at expense of disk I/O. PNG is always lossless
            if cv.imwrite(filename, img, [cv.IMWRITE_PNG_COMPRESSION, 5])==False:
                raise IOError("Failed to save image")
            print("Save image took {:.2f} seconds".format(time.perf_counter() - start_time))
        else:
            print("Skipping",path)
