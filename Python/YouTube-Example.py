import cv2 as cv
import numpy as np


#Grab the image from camera (or a file in this case)
img = cv.imread("grab.jpg")

# Mirror image so sproket hole is on left
img = cv.flip(img,0)

# Mask same size as original image
mask = np.zeros(img.shape[:2], dtype="uint8")

# Draw rectange we want to keep
cv.rectangle(mask,(70,0),(280,1080),255,-1)

masked = cv.bitwise_and(img,img,mask=mask)


# Blur the image
matrix = (5,9)
masked = cv.GaussianBlur(masked,matrix,0)

# Gray scale
masked = cv.cvtColor(masked,cv.COLOR_BGR2GRAY)

# Threshold
_, threshold = cv.threshold(masked, 240,255,cv.THRESH_BINARY)

cv.imshow("threshold",threshold)

# Contours
contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Find the largest contour
contours = sorted(contours, key=lambda x: cv.contourArea(x),reverse=True)

contour = contours[0]

# Get location of rectangle
rect = cv.minAreaRect(contour)
box = cv.boxPoints(rect)

# convert dimensions to ints
box = np.int0(box)

cv.drawContours(img,[box],0, (200,0,200),8)

#cv.imshow("masked",masked)
cv.imshow("image",img)

cv.waitKey(0)

cv.destroyAllWindows()