#!/usr/bin/env python
# coding: utf-8

# # QuestionOne
# ## ORB

# In[ ]:


#%% adding libraris
import cv2
import numpy as np

#%% Question one

# loading main images...
im_template = cv2.imread('template.jpg')
image = cv2.imread('image.jpg')

# showing main Images...
cv2.imshow('mainTemplate',im_template)
cv2.imshow('mainImage',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# initializing ORB
orb = cv2.ORB_create()
kp_t, des_t = orb.detectAndCompute(im_template, None)
kp_i, des_i = orb.detectAndCompute(image, None)

# matcher to draw matching :)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_t, des_i)
matches = sorted(matches, key = lambda x:x.distance)

# Note You can use all found matches,but if you
# draw all of them,the output would be a real mess!
# so I rather to draw some of them 
# matching_result = cv2.drawMatches(im_template, kp_t, image, kp_i, matches, None, flags=2) -> draws all matches.
matching_result = cv2.drawMatches(im_template, kp_t, image, kp_i, matches[:50], None, flags=2)

# leftKeyPoints & rightKeyPoints & matching 
cv2.imshow('leftKeypoints',cv2.drawKeypoints(im_template,kp_t,None))
cv2.imshow('rightKeypoints',cv2.drawKeypoints(image,kp_i,None))
cv2.imshow("Matching", matching_result)

# saving results ...
cv2.imwrite('leftKeypoints.jpg',cv2.drawKeypoints(im_template,kp_t,None))
cv2.imwrite('rightKeypoints.jpg',cv2.drawKeypoints(image,kp_i,None))
cv2.imwrite('matched.jpg',matching_result)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ## SIFT(AKAZE)

# In[ ]:


import cv2
import numpy as np

# loading main images...
im_template = cv2.imread('template.jpg')
image = cv2.imread('image.jpg')

# showing main Images...
cv2.imshow('mainTemplate',im_template)
cv2.imshow('mainImage',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# initializing AKAZE
akaze = cv2.AKAZE_create()
kp_t, des_t = akaze.detectAndCompute(im_template, None)
kp_i, des_i = akaze.detectAndCompute(image, None)

# matcher to draw matching :)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_t, des_i)
matches = sorted(matches, key = lambda x:x.distance)

# Note You can use all found matches,but if you
# draw all of them,the output would be a real mess!
# so I rather to draw some of them 
# matching_result = cv2.drawMatches(im_template, kp_t, image, kp_i, matches, None, flags=2) -> draws all matches.
matching_result = cv2.drawMatches(im_template, kp_t, image, kp_i, matches[:50], None, flags=2)

# leftKeyPoints & rightKeyPoints & matching 
cv2.imshow('leftKeypoints',cv2.drawKeypoints(im_template,kp_t,None))
cv2.imshow('rightKeypoints',cv2.drawKeypoints(image,kp_i,None))
cv2.imshow("Matching", matching_result)

# saving results ...
cv2.imwrite('leftKeypoints.jpg',cv2.drawKeypoints(im_template,kp_t,None))
cv2.imwrite('rightKeypoints.jpg',cv2.drawKeypoints(image,kp_i,None))
cv2.imwrite('matched.jpg',matching_result)

cv2.waitKey(0)
cv2.destroyAllWindows()


# # QuestionTwo
# ## 1-1.JPG

# In[ ]:


import cv2
import numpy as np

# loading and drawing images
img = cv2.imread('1-1.JPG')
cv2.imshow('1-1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# resizedScales
scale_percent = 50 # percent of original size
height,width = int(img.shape[0] * scale_percent / 100),int(img.shape[1] * scale_percent / 100)
dim = (width, height)

# resizing
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
points = np.zeros((4,2),np.int)

# source and destination 
src_points = np.array([
       [470, 490],
       [800, 540],
       [360, 590],
       [760, 670]
])

dst_points = np.array([
       [300,100],
       [300,450],
       [50,100],
       [50,450]
])

# finding homographyMatrix
M,mask = cv2.findHomography(src_points, dst_points)
dst = cv2.warpPerspective(img,M,(700, 600))

# drawing
cv2.imwrite('TopView.jpg',dst)
cv2.imshow("TopView.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## 1-2.JPG

# In[ ]:


import cv2
import numpy as np

# loading and drawing images
img = cv2.imread('1-2.JPG')
cv2.imshow('1-1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# resizedScales
scale_percent = 50 # percent of original size
height,width = int(img.shape[0] * scale_percent / 100),int(img.shape[1] * scale_percent / 100)
dim = (width, height)

# resizing
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
points = np.zeros((4,2),np.int)

# source and destination 
src_points = np.array([
       [540, 390],
       [300, 420],
       [350, 680],
       [720, 600]
])
    
dst_points = np.array([
       [0,0],
       [0,300],
       [600,300],
       [600,0]
])

# finding homographyMatrix
M, mask = cv2.findHomography(src_points, dst_points)
dst = cv2.warpPerspective(img,M,(650,325))
cv2.imwrite('TopView.jpg',dst)

# drawing
cv2.imshow("TopView.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Question Three
# ## 2-1.jpg

# In[ ]:


import cv2
import numpy as np
# flags & counters.
counter = 0
counter_temp = 0

# loading and drawing images
img = cv2.imread('2-1.jpg')
img_original = cv2.imread('2-1.jpg')
wrapper = cv2.imread('2-3.jpg')

# resizedScales
scale_percent = 30 # percent of original size
height,width = int(img.shape[0] * scale_percent / 100),int(img.shape[1] * scale_percent / 100)
dim = (width, height)

# resizing
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img_original = cv2.resize(img_original, dim, interpolation = cv2.INTER_AREA)
points = np.zeros((4,2),np.int)

def mousePoints(event,x,y,flags,params):
    global counter #we need to use it as counter in every parts
    if event == cv2.EVENT_LBUTTONDOWN:
        points[counter] = x,y
        counter = counter + 1
        print(points)
        

finalImage = np.zeros(img.shape,np.uint8)
cv2.namedWindow("image")
cv2.setMouseCallback("image", mousePoints)
    
while True:
    if counter == 4:
        # frame is selected,close mainImage
        if counter_temp ==0:
            cv2.destroyAllWindows()
            counter_temp = 1
        
        # getting configs
        height, width,_ = img.shape
        height_t, width_t,_ = wrapper.shape
        
        # getting source and destination frames
        pts1 = np.float32([points[0],points[1],points[2],points[3]])
        pts2 = np.float32([[0,0],[width_t,0],[0,height_t],[width_t,height_t]])
        
        # getting transformMatrix
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
        # positioning Wrapper in the frame.
        overlay = cv2.warpPerspective(wrapper,matrix,(width,height))
        
        # filling the FinalResult
        for i in range(height):
            for j in range(width):
                if overlay[i,j,0] == 0:
                    finalImage[i,j,:]  = img_original[i,j,:]
                else:
                    finalImage[i,j,:]  = overlay[i,j,:]
                   
       
        cv2.imshow("image", finalImage)               
        
    # drawing Circles
    for x in range (0,counter):
        cv2.circle(img,(points[x][0],points[x][1]),3,(0,255,0),cv2.FILLED)
    if counter != 4:
        cv2.imshow("image", img)
        
    k = cv2.waitKey(1)
    if k == ord('e'):
        break
        
cv2.imwrite('mainImage.jpg',img)
cv2.imwrite('Wrapped.jpg',finalImage)

cv2.destroyAllWindows()


# ## 2-2.jpg

# In[ ]:


import cv2
import numpy as np
# flags & counters.
counter = 0
counter_temp = 0

# loading and drawing images
img = cv2.imread('2-2.jpg')
img_original = cv2.imread('2-2.jpg')
wrapper = cv2.imread('2-3.jpg')

# resizedScales
scale_percent = 30 # percent of original size
height,width = int(img.shape[0] * scale_percent / 100),int(img.shape[1] * scale_percent / 100)
dim = (width, height)

# resizing
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img_original = cv2.resize(img_original, dim, interpolation = cv2.INTER_AREA)
points = np.zeros((4,2),np.int)

def mousePoints(event,x,y,flags,params):
    global counter #we need to use it as counter in every parts
    if event == cv2.EVENT_LBUTTONDOWN:
        points[counter] = x,y
        counter = counter + 1
        print(points)
        

finalImage = np.zeros(img.shape,np.uint8)
cv2.namedWindow("image")
cv2.setMouseCallback("image", mousePoints)
    
while True:
    if counter == 4:
        # frame is selected,close mainImage
        if counter_temp ==0:
            cv2.destroyAllWindows()
            counter_temp = 1
        
        # getting configs
        height, width,_ = img.shape
        height_t, width_t,_ = wrapper.shape
        
        # getting source and destination frames
        pts1 = np.float32([points[0],points[1],points[2],points[3]])
        pts2 = np.float32([[0,0],[width_t,0],[0,height_t],[width_t,height_t]])
        
        # getting transformMatrix
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
        # positioning Wrapper in the frame.
        overlay = cv2.warpPerspective(wrapper,matrix,(width,height))
        
        # filling the FinalResult
        for i in range(height):
            for j in range(width):
                if overlay[i,j,0] == 0:
                    finalImage[i,j,:]  = img_original[i,j,:]
                else:
                    finalImage[i,j,:]  = overlay[i,j,:]
                   
       
        cv2.imshow("image", finalImage)               
        
    # drawing Circles
    for x in range (0,counter):
        cv2.circle(img,(points[x][0],points[x][1]),3,(0,255,0),cv2.FILLED)
    if counter != 4:
        cv2.imshow("image", img)
        
    k = cv2.waitKey(1)
    if k == ord('e'):
        break
        
cv2.imwrite('mainImage.jpg',img)
cv2.imwrite('Wrapped.jpg',finalImage)

cv2.destroyAllWindows()


# # QuestionFour

# In[ ]:


import cv2
import numpy as np

def thresholding(matches,threshold):
    selected = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            selected.append(m)
    return selected

# removing Black parts
def cropping(frame):
    
    # if "np.sum(frame[sth:] or frame[:sth])" is zero so this line
    # is completely Black so lets remove it.
    
    # top horizontal line
    if not np.sum(frame[0]):
        #removingBlackPart
        return cropping(frame[1:])
    
    # bottom horizontal line
    if not np.sum(frame[-1]):
        return cropping(frame[:-2])

    # left vertical line
    if not np.sum(frame[:,0]):
        return cropping(frame[:,1:])

    # right vertical line
    if not np.sum(frame[:,-1]):
        return cropping(frame[:,:-2])
    
    return frame
        
# loading and drawing images
img2 = cv2.imread('3-1.jpeg')
img1 = cv2.imread('3-2.jpeg')

# resizedScales
scale_percent = 80 # percent of original size
height,width =  int(img1.shape[0] * scale_percent / 100),int(img1.shape[1] * scale_percent / 100)
dim = (width, height)

# resizing the images
img1= cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

# initializing AKAZE
akaze = cv2.AKAZE_create()
kpts1, desc1 = akaze.detectAndCompute(img1, None)
kpts2, desc2 = akaze.detectAndCompute(img2, None)

# leftKeyPoints & rightKeyPoints
cv2.imshow('leftKeypoints',cv2.drawKeypoints(img1,kpts1,None))
cv2.imshow('rightKeypoints',cv2.drawKeypoints(img2,kpts2,None))
# writing key Values 
cv2.imwrite('leftKeypoints.jpg',cv2.drawKeypoints(img1,kpts1,None))
cv2.imwrite('rightKeypoints.jpg',cv2.drawKeypoints(img2,kpts2,None))
cv2.waitKey(0)
cv2.destroyAllWindows()

# matcher to draw matching :)
match = cv2.BFMatcher()
matches = match.knnMatch(desc2,desc1,k=2)
    
# adding threshold to KeyPoints
selected = thresholding(matches,0.35)
        
# setting Marker with red Colour for drawing Matches
draw_params = dict(matchColor = (255,0,0),singlePointColor = None,flags = 2)
img3 = cv2.drawMatches(img2,kpts2,img1,kpts1,selected,None,**draw_params)
# drawing matches 
cv2.imshow("Matches", img3)
# saving matches
cv2.imwrite("Matches.jpg", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

# reverse connecting
matches = match.knnMatch(desc1,desc2,k=2)
selected = thresholding(matches,0.35)

# source & dest Keypoints in pictures(among thresholded matches)
src_pts = np.float32([ kpts1[m.queryIdx].pt for m in selected ]).reshape(-1,1,2)
dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in selected ]).reshape(-1,1,2)

# finding homographyMatrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
# warping ...
dst = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], 2*img2.shape[0]))
dst[0:img2.shape[0],0:img2.shape[1]] = img2


cv2.imshow("stitchedCrop", cropping(dst))
cv2.imwrite("stitchedCrop.jpg", cropping(dst))

cv2.waitKey(0)
cv2.destroyAllWindows()


# # QuestionFive
# ## part 1

# In[3]:


import cv2
import numpy as np

# loading images
img1 = cv2.imread('4-1.jpg')
img2 = cv2.imread('4-2.jpg')

def drawLeftCorners(img):
    # finding chessBoardCorners
    retval,chessBoardCorners = cv2.findChessboardCorners(img,(7,8))
    numOfCorners = len(chessBoardCorners)

    result = np.ndarray.copy(img)
    # drawing
    for i in range(numOfCorners):
        points = chessBoardCorners[i]
        result = cv2.circle(result,(points[0][0],points[0][1]),3,(0,0,255),cv2.FILLED)
    return result,chessBoardCorners


def drawRightCorners(img):
    # finding chessBoardCorners
    retval,chessBoardCorners = cv2.findChessboardCorners(img,(6,5))
    numOfCorners = len(chessBoardCorners)

    result = np.ndarray.copy(img)
    # drawing
    for i in range(numOfCorners):
        points = chessBoardCorners[i]
        result = cv2.circle(result,(points[0][0],points[0][1]),3,(0,0,255),cv2.FILLED)
    return result,chessBoardCorners



img1Left,pts1Left = drawLeftCorners(img1)
img1Right,pts1Right = drawRightCorners(img1)

img2Left,pts2Left = drawLeftCorners(img2)
img2Right,pts2Right = drawRightCorners(img2)

cv2.imshow("FirstImageLeftCorners.jpg", img1Left)
cv2.imshow("FirstImageRightCorners.jpg", img1Right)

cv2.imwrite("FirstImageLeftCorners.jpg", img1Left)
cv2.imwrite("FirstImageRightCorners.jpg", img1Right)

cv2.imshow("SecondImageLeftCorners.jpg", img2Left)
cv2.imshow("SecondImageRightCorners.jpg", img2Right)

cv2.imwrite("SecondImageLeftCorners.jpg", img2Left)
cv2.imwrite("SecondImageRightCorners.jpg", img2Right)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ## part 2,3
# ### warning: before running you should first run part1.

# In[4]:



firstImageAllCorners = np.row_stack((pts1Left,pts1Right))
secondImageAllCorners = np.row_stack((pts2Left,pts2Right))

#finding fundamental matrix
fundamental = cv2.findFundamentalMat(firstImageAllCorners,
                                     secondImageAllCorners,
                                     cv2.FM_LMEDS)

F = fundamental[0]

# extracting epipolar line with stereo
img3 = cv2.imread('4-3.jpg')
img4 = cv2.imread('4-4.jpg')
consideredPoint = np.array([265,305])
lines = cv2.computeCorrespondEpilines(consideredPoint.reshape(-1,1,2),2,F)

# this function is extracted from StackOverFlow :)
# https://stackoverflow.com/questions/51089781/
# how-to-calculate-an-epipolar-line-with-a-stereo-pair-of-images-in-python-opencv

def drawLines(img1,img2,lines,pts2):
    _,c,_ = img1.shape
    r = lines.reshape(3)
    color = (0,0,255)
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
    img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    img2 = cv2.circle(img2,tuple(pts2),5,color,-1)
    return img1,img2

img3,img4 = drawLines(img3,img4,lines,consideredPoint)

cv2.imshow("Image4",img4)
cv2.imshow("Image3Lined", img3)
cv2.imwrite("Image4.jpg", img4)
cv2.imwrite("Image3Lined.jpg", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ## part 4
# ### warning: before running you should first run part2,3.

# In[30]:


from random import randrange
import cv2
import numpy as np

img3 = cv2.imread('4-3.jpg')
img4 = cv2.imread('4-4.jpg')

counter = 10

def drawLines(img1,img2,lines,pts2):
    global counter
    _,c,_ = img1.shape
    r = lines.reshape(3)
    color = (counter,
             randrange(counter + 1),
             255 - randrange(counter%255 + 1))
    
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
    img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    img2 = cv2.circle(img2,tuple(pts2),5,color,-1)
    counter = (counter + 10 )%255
    return img1,img2

# it draws 4-3's lines in 4-4
# change firstImageAllCorners to secondImageAllCorners
# to draw 4-4's lines in 4-3.

for i in range(len(firstImageAllCorners)): # or len(secondImageAllCorners),both are 86.
    point = firstImageAllCorners[i,:,:] 
    lines = cv2.computeCorrespondEpilines(point.reshape(-1,1,2)
                                          ,1,F)
    img4,img3 = drawLines(img4,img3,lines,point.reshape(2))

cv2.imshow("Image3", img3)
cv2.imshow("Image4Lined", img4)
cv2.imwrite("Image4Lined.jpg", img4)
cv2.imwrite("Image3.jpg", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

