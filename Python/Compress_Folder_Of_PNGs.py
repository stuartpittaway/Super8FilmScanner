# Simple script to copy images from one folder to another
# converting/compressing PNG along the way
# to aid in reducing disk space and reducing CPU load on Rasp PI
import numpy as np
import cv2 as cv
import time
import glob
import os

def Filelist(path: str, ext: str) -> int:
    return sorted(glob.glob(os.path.join(path, "frame_????????."+ext)), reverse=True)

output_path = os.path.join(os.getcwd(), "Capture")
if not os.path.exists(output_path):
    raise FileNotFoundError("Missing output folder")

input_path = "\\\\192.168.0.60\\pi\\Super8FilmScanner\\Python\\Capture-8.0"
if not os.path.exists(input_path):
    raise FileNotFoundError("Missing input folder")

while True:
    # Scan for bitmap files to convert, skip first one (most recent) in cas its being written to
    for filename in Filelist(input_path,"bmp")[2:]:
        img = cv.imread(filename,cv.IMREAD_UNCHANGED)
        if img is None:
            print("Error opening file",filename)
        else:
            output_filename=os.path.join(output_path,os.path.splitext(os.path.basename(filename))[0]+'.png')
        
            start_time=time.perf_counter()
            # PNG output, with NO compression - which is quicker (less CPU time) on Rasp PI
            # at expense of disk I/O. PNG is always lossless
            if cv.imwrite(output_filename, img, [cv.IMWRITE_PNG_COMPRESSION, 2])==True:
                print("Save image took {:.2f} seconds".format(time.perf_counter() - start_time))
                #Delete source file
                os.remove(filename)
                print("Deleted",filename)
            else:
                raise IOError("Failed to save image")

    time.sleep(20)
