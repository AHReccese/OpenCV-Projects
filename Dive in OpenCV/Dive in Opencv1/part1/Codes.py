import numpy as np
import cv2
# Question 3 
img = cv2.imread('2.jpg',0)
width = 400 # j range
height = 300 # i range

dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

x_edge = np.zeros(img.shape,np.float)
y_edge = np.zeros(img.shape,np.float)

edge = np.zeros(img.shape,np.float)
mean = np.zeros(img.shape,np.float)

img = img.astype(dtype=np.float)

# y edgeing...
for i in range(height):
    for j in range(width):
        if(j == width-1):
            y_edge[i,j] = -1*img[i,j] # it is zero padded : img[i,j+1] = 0 & img[i,j+1] - img[i,j] = -img[i,j]
        else:
            y_edge[i,j] = img[i,j+1] - img[i,j] # forward derivation :y
            
# x edgeing
for i in range(height):
    for j in range(width):
        if(i == height-1):
            x_edge[i,j] = -1* img[i,j] # it is zero padded : img[i+1,j] = 0 & img[i+1,j] - img[i,j] = -img[i,j]
        else:
            x_edge[i,j] = img[i+1,j] - img[i,j] # forward derivation :x
            
# meaning!!!
for i in range(height):
    for j in range(width):
        
      
        if i == 0:
            if j == 0:
            # img[0,0] -> meaning 4 pixels.
                mean[i,j] = img[i,j] + img[i,j+1] + img[i+1,j] + img[i+1,j+1]
            elif j == width-1:
            #img[i,399] -> meaning 4 pixels.
                mean[i,j] = img[i,j] + img[i,j-1] + img[i+1,j-1] + img[i+1,j]
            else:
                mean[i,j] = img[i,j-1] + img[i+1,j-1] + img[i,j] + img[i+1,j] + img[i,j+1] + img[i+1,j+1]
           
        
        elif i == height-1: 
            if j == 0:
                mean[i,j] = img[i,j] + img[i-1,j] + img[i,j+1] + img[i-1,j+1]
            elif j == width -1:
                mean[i,j] = img[i,j] + img[i-1,j] + img[i-1,j-1] + img[i,j-1]
            else:
                mean[i,j] = img[i,j] + img[i,j-1] + img[i,j+1] + img[i-1,j-1] + img[i-1,j] + img[i-1,j+1]
        
        
        
        elif j == 0:
            mean[i,j] = img[i,j] + img[i-1,j] + img[i+1,j] + img[i-1,j+1] + img[i,j+1] + img[i+1,j+1]
            
        elif j == width -1:
            mean[i,j] = img[i,j] + img[i-1,j] + img[i+1,j] + img[i-1,j-1] + img[i,j-1] + img[i+1,j-1]
            
        else:
            mean[i,j] = img[i-1,j-1] + img[i-1,j] + img[i-1,j+1]
            + img[i,j-1]   + img[i,j]   + img[i,j+1]
            + img[i+1,j-1] + img[i+1,j] + img[i+1,j+1]
            
# getting average.            
mean = (1/9)*mean

# high frequency Filter.
# edge = sqrt(edge_x^2 + edge_y^2)
for i in range (height):
    for j in range (width):
        edge[i,j] = (x_edge[i,j]**2 + y_edge[i,j]**2)**(1/2)
        
y_edge = np.absolute(y_edge)
x_edge = np.absolute(x_edge)

y_edge = y_edge.astype(dtype=np.uint8)
x_edge = x_edge.astype(dtype=np.uint8)

img = img.astype(dtype=np.uint8)
mean = mean.astype(dtype=np.uint8)
edge = edge.astype(dtype=np.uint8)


cv2.imshow('yEdge',y_edge)
cv2.imshow('xEdge',x_edge)

# img : mainImage
# Upfrequecny : mean
# you can consider Upfrequency to : img - mean

cv2.imshow('UpFrequency',edge)
cv2.imshow('LowPass',mean)
cv2.imshow('MainImage',img)

upfreqFromLowPass = img - mean
cv2.imshow('UpfreqFromLowPass',upfreqFromLowPass)

cv2.waitKey(0)
cv2.destroyAllWindows()




#Question 3 -> Mean kernel! 
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('2.jpg')
img = cv2.imread('2.jpg',0)
width = 400 # j range
height = 300 # i range
kernelDim = 7
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

kernel = np.ones((kernelDim,kernelDim),np.float32)/(kernelDim**2)
dst = cv2.filter2D(img,-1,kernel)
cv2.imshow('LossPassViaMeanKernel',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Question 4 -> Canny 
import tkinter as tk
from tkinter import simpledialog
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Loading images
img1 = cv2.imread('1.jpg',0)
img2 = cv2.imread('2.jpg',0)

ROOT = tk.Tk()
ROOT.withdraw()

# getting scale
scale = int(simpledialog.askstring(title="Getting Scale", prompt="Scale"))
print("Scale: ", scale)

width = int(img1.shape[1] * scale / 100)
height = int(img1.shape[0] * scale / 100)
dim = (width, height)
# resize image
img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

# the input dialog
minVal = int(simpledialog.askstring(title="Getting MinVal", prompt="minVal"))
maxVal = int(simpledialog.askstring(title="Getting MaxVal", prompt="maxVal"))

# check it out
print("MinValue: ", minVal)
print("MaxValue: ", maxVal)

canny1 = cv2.Canny(img1,minVal,maxVal)
canny2 = cv2.Canny(img2,minVal,maxVal)
cv2.imshow('canny',canny1)
cv2.imshow('canny2',canny2)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Question 4 -> Sobel
import cv2 

def sobel(image,scale):
    
    delta = 0
    ddepth = cv2.CV_16S

    image = cv2.GaussianBlur(image, (3, 3), 0)
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)    
    
    return grad
    
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
scale = 0.75

sobel1 = sobel(img1,scale)
sobel2 = sobel(img2,scale)

cv2.imshow('sobel(1.jpg)',sobel1)
cv2.imshow('sobel(2.jpg)',sobel2)


cv2.waitKey(0)
cv2.destroyAllWindows()



# Question 4 -> loG
import cv2 
import numpy as np

def loG(image):
    # laplacian of gaussian methoded
    # applying gaussian filter    
    # converting to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)
    # convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

loG1 = loG(img1)
loG2 = loG(img2)

cv2.imshow('loG(1.jpg)',loG1)
cv2.imshow('loG(2.jpg)',loG2)

cv2.waitKey(0)
cv2.destroyAllWindows()



# blob detection Question 6
import cv2
import numpy as np
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()


# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 300;

# Filter by Area.
params.filterByArea = True
params.minArea = 150

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.01

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01


detector = detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
image = cv2.imread('3.jpg')
keypoints = detector.detect(image)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()




#section B Question 1 -> recoding video
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('RecordedVideo.avi',fourcc, 20.0, (640,480))
record = False
while(cap.isOpened()):
    ret, frame = cap.read()
    # write the flipped frame
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    
    if((key)and(key == ord('s'))):
        record = True
    elif((key) and (key == ord('e'))):
        break
    if(record):
        out.write(frame)
     
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()





# Question 3 of part b
# without gaussian filter 

import cv2
import numpy as np


def sobel(image,scale):
    
    delta = 0
    ddepth = cv2.CV_16S

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)    
    
    return grad


def prewitt(image):
    
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    prewitt = prewittx + prewitty
    return prewitt


cap = cv2.VideoCapture('RecordedVideo.avi')
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
framenum = cap.get(cv2.CAP_PROP_POS_FRAMES)
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
outSobel = cv2.VideoWriter('Sobel.avi',fourcc, 20.0, (640,480),False)

#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
outCanny = cv2.VideoWriter('Canny.avi',fourcc, 20.0, (640,480),False)

#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
outPrewitt = cv2.VideoWriter('Prewitt.avi',fourcc, 20.0, (640,480),False)

outAll = cv2.VideoWriter('Comparison.avi',fourcc, 20.0, (2*640,2*480),False)


for i in range(int(framenum)):
    ret,frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # filters
    canny = cv2.Canny(gray,40,90)
    sobelImage = sobel(frame,1)
    prewit = prewitt(gray)
    
    Up = cv2.hconcat([gray,canny])
    Down = cv2.hconcat([sobelImage,prewit])
    All = cv2.vconcat([Up,Down])
    
    outSobel.write(sobelImage)
    outCanny.write(canny)
    outPrewitt.write(prewit)
    outAll.write(All)
    cv2.imshow('sobel',sobelImage)
    cv2.imshow('canny',canny)
    cv2.imshow('prewitt',prewit)
        
cap.release()
outSobel.release()
outCanny.release()
outPrewitt.release()
outAll.release()
cv2.destroyAllWindows()




# Question 3 of part b
# with gaussian filter 
# comparison of Methods.

import cv2
import numpy as np


def sobel(image,scale):
    
    delta = 0
    ddepth = cv2.CV_16S
     
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)    
    
    return grad


def prewitt(image):
    
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    prewitt = prewittx + prewitty
    return prewitt


cap = cv2.VideoCapture('RecordedVideo.avi')
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
framenum = cap.get(cv2.CAP_PROP_POS_FRAMES)
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
outSobel = cv2.VideoWriter('SobelVsNoisyComparison.avi',fourcc, 20.0, (640,2*480),False)

outCanny = cv2.VideoWriter('CannyVsNoisyComparison.avi',fourcc, 20.0, (640,2*480),False)

outPrewitt = cv2.VideoWriter('PrewittVsNoisyComparison.avi',fourcc, 20.0, (640,2*480),False)

for i in range(int(framenum)):
    ret,frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayGaussFilter = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    # filters
    canny = cv2.Canny(gray,40,90)
    cannyGaussFilter = cv2.Canny(grayGaussFilter,40,90)
    outcanny = cv2.vconcat([canny,cannyGaussFilter])
    
    sobelImage = sobel(gray,1)
    sobelImageGaussFilter = sobel(grayGaussFilter,1)
    outsobel = cv2.vconcat([sobelImage,sobelImageGaussFilter])
    
    prewit = prewitt(gray)
    prewitGaussFilter = prewitt(grayGaussFilter)
    outprewit = cv2.vconcat([prewit,prewitGaussFilter])
    
    outSobel.write(outsobel)
    outCanny.write(outcanny)
    outPrewitt.write(outprewit)

cap.release()
outSobel.release()
outCanny.release()
outPrewitt.release()
cv2.destroyAllWindows()




