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

# These are automatically calculated if not supplied, based on the first image in the folder
#target_w=2650
#target_h=1790

# Size of images on screen (scaled down from 1920x1080)
PREVIEW_SCALE=0.5

for filename in files:
    img = cv.imread(filename,cv.IMREAD_UNCHANGED)
    if img is None:
        raise Exception("Error opening file",filename)
    else:

        # The first frame is our default image size, make all other images this size
        if target_h==None or target_w==None:
            target_h, target_w =img.shape[:2]

        h, w =img.shape[:2]

        if target_h!=h or target_w!=w:
            print("Resize the image, wrong dimensions")
            new_image = np.zeros((target_h,target_w,3), np.uint8)
            new_image[0:h,0:w]=img.copy()
            img=new_image.copy()
            new_image=None
            h, w =img.shape[:2]


        # Resize image to Full HD 1920x1080 and put into 16:9 frame?
        # Comment this out to leave at original resolution (note very slow!!)
        if True==True:
            output_w=1920
            output_h=1080

            #Scale new_image to keep correct aspect ratio
            scale = output_w/w
            if h*scale > output_h:
                scale = output_h/h

            scale_w=int(w*scale)
            scale_h=int(h*scale)

            print("Scaled image w=",scale_w,"h=",scale_h, "original w=",w,"h=",h)
            #Horizontal centre frame
            #scale_x_offset=int(output_w/2 - scale_w/2)
            img=cv.resize(img.copy(), (scale_w,scale_h), interpolation=cv.INTER_AREA)

            #Now place on a 1920x1080 frame
            #new_image = np.zeros((output_h,output_w,3), np.uint8)
            #new_image[0:scale_h,scale_x_offset:scale_x_offset+scale_w]=scaled_image
            #img=new_image.copy()
            #new_image=None

        # Debug resize to speed up processing
        #img = cv.resize(img, (0,0), fx=0.3, fy=0.3)

        # Update the dimensions, as they could have changed by now
        h, w =img.shape[:2]

        frames.append(img)
        if len(frames)!=3:
            continue

        #Our first output frame is number 1 (frame zero is skipped, as is the last one)
        #This will renumber the frames if the input doesn't start at zero
        frame_number+=1
        output_filename = os.path.join(output_path, "frame_{:08d}.png".format(frame_number))

        print(os.path.basename(filename),output_filename)
        if os.path.exists(output_filename):
            print("Skip file",filename)
        else:
            print("Processing...")
            
            # Frames has 0,1,2 image array
            dst = cv.fastNlMeansDenoisingColoredMulti(frames, 1, 1, dst=None, h=3.5, hColor=3.5, templateWindowSize=7,searchWindowSize=21)

            if cv.imwrite(output_filename, dst, [cv.IMWRITE_PNG_COMPRESSION, 2])==False:
                raise IOError("Failed to save image")

            cv.imshow("image_orig",cv.resize(frames[1], (0,0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE))
            cv.imshow("image_fixed",cv.resize(dst, (0,0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE))
            #cv.imshow("image_orig",frames[1])
            #cv.imshow("image_fixed",dst)
            cv.moveWindow("image_orig",0,80)
            cv.moveWindow("image_fixed",int(w*PREVIEW_SCALE),80)

            # Test for key and allow screen to refresh
            k = cv.waitKey(250) & 0xFF

            if k == 27:
                break

        # Remove oldest image
        frames.pop(0)


#Exit
cv.destroyAllWindows()
