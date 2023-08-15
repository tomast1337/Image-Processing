# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:47:31 2023

@author: Nicolas Vycas Nery
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('Pictures/arroz_e_feijao.webp')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
"""
#or
gray = []
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        gray[i,j] = image[i][j].sum() // 3

gray = np.array(gray)
"""


# Convert to Black and White
(thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
"""
#or
blackAndWhiteImage = []
upper = 255
lower = 0
threshold = 127 # 255//2
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        if image[i][j].sum() // 3 > threshold:
            blackAndWhiteImage[i,j] = upper
        else:
            blackAndWhiteImage[i,j] = lower

blackAndWhiteImage = np.array(blackAndWhiteImage)
"""
# Show the images

cv2.imshow('Original image',image)
cv2.waitKey(0)

cv2.imshow('Gray image', gray)
cv2.waitKey(0)

# create a histogram of the image and the gray image

domain = np.arange(0,256)
histGrey = [0]*256
histR = [0]*256
histG = [0]*256
histB = [0]*256

for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        histGrey[gray[i,j]] += 1
        histR[image[i,j,2]] += 1
        histG[image[i,j,1]] += 1
        histB[image[i,j,0]] += 1

histGrey = np.array(histGrey)
histR = np.array(histR)
histG = np.array(histG)
histB = np.array(histB)


# plot a single histogram with all values

plt.figure()
plt.title('Histogram')
plt.xlabel('Bins')
plt.ylabel('number of pixels')
plt.bar(domain,histGrey,color="grey")
plt.bar(domain,histR,color="red")
plt.bar(domain,histG,color="green")
plt.bar(domain,histB,color="blue")
plt.show()

# fill a new array with 255 in the three channels
img_upper_cut = np.zeros(image.shape, np.uint8)
img_lower_cut = np.zeros(image.shape, np.uint8)

for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        # if pixel is more than 150
        if image[i,j].sum() // 3 > 150:
            img_upper_cut[i,j] = image[i,j]
            #img_lower_cut[i,j] = 255
        else:
            img_lower_cut[i,j] = image[i,j]
            #img_upper_cut[i,j] = 255


img_upper_cut = np.array(img_upper_cut)
img_lower_cut = np.array(img_lower_cut)

cv2.imshow('Upper cut',img_upper_cut)
cv2.waitKey(0)

cv2.imshow('Lower cut',img_lower_cut)
cv2.waitKey(0)

        