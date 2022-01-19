# Simple script to copy images from one folder to another
# converting/compressing PNG along the way
# to aid in reducing disk space and reducing CPU load on Rasp PI
import numpy as np
import cv2 as cv
import time
import glob
import os

def Filelist(path: str, ext: str) -> int:
    return sorted(glob.glob(os.path.join(path, "frame_????????."+ext)), reverse=False)

input_path = "E:\\source\\Super8FilmScanner\\Python\\Aligned"
if not os.path.exists(input_path):
    raise FileNotFoundError("Missing input folder")

output_path = os.path.join(os.getcwd(), "Denoise")
if not os.path.exists(output_path):
    raise FileNotFoundError("Missing output folder")

files=Filelist(input_path,"png")
if len(files)==0:
    print("Nothing to do, so quit...")
    quit()

frames=[]
frame_number=0
target_h=None
target_w=None

for filename in files:
    img = cv.imread(filename,cv.IMREAD_UNCHANGED)
    if img is None:
        raise Exception("Error opening file",filename)
    else:

        # Debug resize to speed up processing
        #img = cv.resize(img, (0,0), fx=0.3, fy=0.3)

        # The first frame is our default image size, make all other images this size
        if target_h==None or target_w==None:
            target_h, target_w =img.shape[:2]

        h, w =img.shape[:2]

        if target_h!=h or target_w!=w:
            print("Resize the image")
            new_image = np.zeros((target_h,target_w,3), np.uint8)
            new_image[0:h,0:w]=img.copy()
            img=new_image.copy()
            new_image=None

        frames.append(img)
        if len(frames)!=3:
            continue

        #Our first output frame is number 1 (frame zero is skipped, as is the last one)
        frame_number+=1
        output_filename = os.path.join(output_path, "frame_{:08d}.png".format(frame_number))

        print(os.path.basename(filename),output_filename)
        if os.path.exists(output_filename):
            print("Skip file",filename)
        else:
            print("Processing...")
            
            # Frames has 0,1,2 image array
            dst = cv.fastNlMeansDenoisingColoredMulti(frames, 1, 1, dst=None, h=3, hColor=3, templateWindowSize=7,searchWindowSize=29)

            if cv.imwrite(output_filename, dst, [cv.IMWRITE_PNG_COMPRESSION, 2])==False:
                raise IOError("Failed to save image")

            cv.imshow("image_orig",cv.resize(frames[1], (0,0), fx=0.25, fy=0.25))
            cv.imshow("image_fixed",cv.resize(dst, (0,0), fx=0.25, fy=0.25))
            #cv.imshow("image_orig",frames[1])
            #cv.imshow("image_fixed",dst)
            cv.moveWindow("image_orig",0,80)
            cv.moveWindow("image_fixed",int(target_w*0.25),80)

            k = cv.waitKey(100) & 0xFF
           
            if k == 27:
                break

        # Remove oldest image
        frames.pop(0)


#Exit
cv.destroyAllWindows()
