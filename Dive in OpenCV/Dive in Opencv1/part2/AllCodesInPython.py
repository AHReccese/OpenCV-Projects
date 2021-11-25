import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
print('Importing Libraries')

####

import cv2 as cv
# sample read & show.
img1 = cv.imread('space.jpg')
cv.imshow('img1',img1)
cv.waitKey(0)
cv.destroyAllWindows()

####

import cv2 as cv
# Question 1 part a adding text
img1 = cv.imread('1.jpg')
# loading image with color
font = cv.FONT_HERSHEY_COMPLEX
string = "96101635"
cv.putText(img1,string,(0,10),font,.5,(0,0,0),2,cv.LINE_AA)
cv.imshow('img1',img1)
gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imshow('gray_image',gray_img)
while(1):
    key = cv.waitKey(0)
    if (key == ord('s')):
        cv.imwrite('RGB_sample.jpg',img1)
        cv.imwrite('grayscale_sample.jpg',gray_img)
        cv.destroyAllWindows()
        break
    elif (key == ord('e')):
        cv.destroyAllWindows()
        break
		
####

# run time typing on image ...
# i did it voluntarily
img = cv.imread('1.jpg')
# initialize counter
i = 0
# run time typing.
while True:
    # Display the image
    cv.imshow('voluntarilyPart',img)
    # wait for keypress
    k = cv.waitKey(0)
    # specify the font and draw the key using puttext
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img,chr(k),(i,10), font,.5,(0,0,0),2,cv.LINE_AA)
    i+=10
    if k == ord('q'):
        break
cv.destroyAllWindows()

####

# Question 1 part b finding ball and add another one
# import the necessary packages
import numpy as np
import cv2

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread('football.jpg')
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect circles in the image
# special setting: in order to extract the ball.
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 100, 
               param2 = 45, minRadius = 10, maxRadius = 40) 

# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    print(len(circles))
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        
        #showing the found circle.
        #cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        
        cv2.rectangle(output, (x - r-3, y - r-3), (x + r + 3, y + r + 3), (0, 255, 0), 3)
        cv.imwrite('rected.jpg',output)
        newx = x + 10*r
        newy = y
        output[newy-r-3:newy+r+3,newx-r-3:newx+r+3] = image[y-r-3:y+r+3,x-r-3:x+r+3]
    # show the output image
    result = np.hstack([image, output])
    cv2.imshow("output",result)
    cv.imwrite('BallAdded.jpg',output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

####


import cv2
import numpy as np
import math
#%% Question 2

def callback(x):
    return

def merge(left,right):
    return cv2.hconcat([left ,right])
    

def rotate_image_size_corrected(image, angle):
    # Calculate max size for the rotated template and image offset
    image_size_height, image_size_width,noteUsed = image.shape
    image_center_x = image_size_width / 2
    image_center_y = image_size_height / 2

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((image_center_x, image_center_y), angle, 0.8)
    # Apply rotation to the template
    image_rotated = cv2.warpAffine(image, rotation_matrix, (image_size_width, image_size_height))
    return [image_rotated,rotation_matrix]



def scale(mat,percent):
    if(len(percent) == 1):
        width = int(mat.shape[1] * percent[0] / 100)
        height = int(mat.shape[0] * percent[0] / 100)
        dim = (width, height)
        # resize image
        return cv2.resize(mat, dim, interpolation = cv2.INTER_AREA)
    else:
        width = int(mat.shape[1] * percent[1] / 100)
        height = int(mat.shape[0] * percent[0] / 100)
        dim = (width, height)
        # resize image
        return cv2.resize(mat, dim, interpolation = cv2.INTER_AREA)
        
    
mainImage = cv2.imread('space.jpg')
cv2.namedWindow('rotationQuestion')
cv2.createTrackbar('Rotate degree','rotationQuestion',0,360,callback)

# resize image
percent = 50
mainImage = scale(mainImage,[percent])

rowMain , colMain , channel = mainImage.shape
#we have second resize for making best worstcase!

# creating rightSide picture
scaled = scale(mainImage,[200,100])
rightrow , rightcol , channel = scaled.shape
updownBlack = np.zeros((int((rightrow - rowMain)/2),rightcol,channel),np.uint8)
upper = cv2.vconcat([updownBlack, mainImage])
rightImage = cv2.vconcat([upper,updownBlack])

beforeRightDot = np.matrix([[30],[30+int((rightrow - rowMain)/2)],[1]])

while(1):
    
    degree = cv2.getTrackbarPos('Rotate degree','rotationQuestion')
    [rotated,R] =rotate_image_size_corrected(rightImage,degree)
    rotated = cv2.resize(rotated,(rightcol,rightrow),interpolation = cv2.INTER_AREA)
    tempDot = R * beforeRightDot
    
    final = merge(scaled ,rotated)
    
    # Adding offsets: findind tempDot place in final picture.
    second_point_x = rightcol + tempDot[0]
    second_point_y = tempDot[1]
    afterRightDot = (second_point_x, second_point_y)
    
    # Drawing on final Result
    final = cv2.line(final, (30,30),afterRightDot, (0,0,255), 2)
    cv2.imshow('rotationQuestion',final)
    key = cv2.waitKey(1)
    if key == ord('e'):
        break;
    
cv2.destroyAllWindows()

####

#Question 5
import cv2 
import numpy as np 

def addthreshold(image,key):
    if key == 0:
        return image
    else:
        return cv2.threshold(img ,90,255,cv2.THRESH_BINARY)
  
 # Reading the input image 
mainImg = cv2.imread('limbo.png', 0) 
  
# one of the best ways to improve efficiency of the following kernels is
# is to fitler the main image from a threshold.
# set key to 1 if you wanna see the efficiency of threshold.
key = 0
img = addthreshold(mainImg,key)

# Taking a matrix of size 5 as the kernel

#training different matrixes
# 3*3 kernel
kernel3_3 = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)
# 5*5 kernel
kernel5_5 = np.ones((5,5), np.uint8) 
# 7*7 kernel
kernel7_7 = np.array([[1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1]],np.uint8)

# The first parameter is the original image, 
# kernel is the matrix with which image is  
# convolved and third parameter is the number  
# of iterations, which will determine  how much  
# you want to erode/dilate a given image.  
# kernel sample 3 is chosen.

img_erosion = cv2.erode(img, kernel7_7, iterations=1) 
img_dilation = cv2.dilate(img, kernel7_7, iterations=1) 
img_opening = cv2.dilate(img_dilation, kernel7_7, iterations=1) 
img_closing = cv2.erode(img_erosion, kernel7_7, iterations=1) 
  
cv2.imshow('origin',mainImg)    
cv2.imshow('dilated',img_dilation)
cv2.imshow('erosed',img_erosion)
cv2.imshow('opened',img_opening)
cv2.imshow('closed',img_closing)
cv2.waitKey(0)
cv2.destroyAllWindows()


from matplotlib import pyplot as plt

plt.figure()
titels = ['dilated','erosed','opened','closed']
pics = [img_dilation,img_erosion,img_opening,img_closing]

plt.subplots(2,2,figsize = (40,40))
for i in range(1,5):
    plt.subplot(2,2,i)
    plt.imshow(pics[i-1],'gray')
    plt.title(titels[i-1],fontsize = 50)
plt.show()

####

# Question 7 shape ditection
import cv2
import numpy as np

gray = cv2.imread('4.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.imread('4.jpg')
rows,cols = gray.shape
th, zero_one = cv2.threshold(gray, 80, 1, cv2.THRESH_BINARY_INV)

kernel = np.zeros((20,30))

#setting borders to 1
# nose detection!!! 20,30 is the nose's dimensions.
kernel[0, :] = 1
kernel[19, :] = 1
kernel[:, 0] = 1
kernel[:, 29] = 1

filtered = cv2.filter2D(zero_one,-1,kernel)

# finding good threshhold
output = cv2.threshold(filtered, 80, 255, cv2.THRESH_BINARY)[1]
#noses numbers -> it also determines (x,y) of the nose's center.
centers = np.argwhere(output == 255)
noOfNoses = centers.shape[0]

# swapping x,y : x,y <= y,x
for i in range(noOfNoses):
    centers[i][0],centers[i][1] = centers[i][1],centers[i][0]
    
for center in centers:
    final = cv2.circle(img, tuple(center), 20, (0, 0, 255))

cv2.imshow('zero_one',zero_one)
cv2.imshow('filtered',filtered)
cv2.imshow('output',output)
cv2.imshow('final',final)
cv2.waitKey(0)
cv2.destroyAllWindows()

####

#section B Question 1
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('RecordedQB_1.avi',fourcc, 20.0, (640,480))
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


####

import cv2
import numpy as np
# section B Question 2 Background reductuction

cap = cv2.VideoCapture('video.mp4')
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
numberOfFrames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

Frames = [None]*numberOfFrames
for i in range(numberOfFrames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    notUsed,frame = cap.read()
    Frames[i] = frame

cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)

#extracting background
Background = np.median(Frames, axis=0).astype(dtype=np.uint8)   
cv2.imshow('frame', Background)
cv2.imwrite('backFrame.jpg',Background)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
len(Frames)
for i in range(len(Frames)):
    print("hi")
    frame = Frames[i]
    diff = cv2.subtract(frame,Background)
    cv2.imshow('frame',diff)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

####	
