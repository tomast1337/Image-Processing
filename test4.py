# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:47:31 2023

@author: Nicolas Vycas Nery
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('img1.jpg')

# rezise the image
scale_percent = 60 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# # resize image
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

#or
"""
gray = np.zeros((image.shape[0],image.shape[1]))
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        gray[i][j] = image[i][j].sum() // 3

gray = np.array(gray)
"""

r1 = 50
r2 = 200

expantion = np.zeros((image.shape[0],image.shape[1]), dtype="uint8")
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        new_color = 255*( (gray[i,j] - r1) / (r2 - r1))
        # ensure the new color is between 0 and 255
        new_color = max(0,min(new_color,255))
            
        expantion[i,j] = new_color


expantion = np.array(expantion)

compresion = np.zeros((image.shape[0],image.shape[1]), dtype="uint8" )
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        compresion[i,j] = r1 + (r2 - r1) * (gray[i,j] / 255)

compresion = np.array(compresion)

r1 = 100
r2 = 150
r3 = 200

s1 = 50
s2 = 100
s3 = 150

expantion2 = np.zeros((image.shape[0],image.shape[1]), dtype="uint8")
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        if gray[i,j] <= r1:
            expantion2[i,j] = 0
        elif gray[i,j] > r1 and gray[i,j] <= r2:
            expantion2[i,j] = s1 + (s2 - s1) * (gray[i,j] / 255)
        elif gray[i,j] > r2 and gray[i,j] <= r3:
            expantion2[i,j] = s2 + (s3 - s2) * (gray[i,j] / 255)
        else:
            expantion2[i,j] = 255

expantion2 = np.array(expantion2)

compresion2 = np.zeros((image.shape[0],image.shape[1]), dtype="uint8")

for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        if gray[i,j] <= s1:
            compresion2[i,j] = 0
        elif gray[i,j] > s1 and gray[i,j] <= s2:
            compresion2[i,j] = r1 + (r2 - r1) * (gray[i,j] / 255)
        elif gray[i,j] > s2 and gray[i,j] <= s3:
            compresion2[i,j] = r2 + (r3 - r2) * (gray[i,j] / 255)
        else:
            compresion2[i,j] = 255

compresion2 = np.array(compresion2)

# Show the images
cv2.imshow('Original image',gray)
cv2.imshow('Expanded image', expantion)
cv2.imshow('Compressed image', compresion)
#show the original image histogram
plt.title('Original image histogram')
plt.hist(gray.ravel(),256,[0,256])
plt.show()
#show the scaled image histogram
plt.title('Scaled image histogram')
plt.hist(expantion.ravel(),256,[0,256])
plt.show()

#show the compressed image histogram
plt.title('Compressed image histogram')
plt.hist(compresion.ravel(),256,[0,256])
plt.show()

cv2.waitKey(0)


cv2.imshow('Original image',gray)
cv2.imshow('Expanded image multi nivel', expantion2)
cv2.imshow('Compressed image multi nivel', compresion2)
#show the original image histogram
#plt.title('Original image histogram')
#plt.hist(gray.ravel(),256,[0,256])
#plt.show()

#show the scaled image histogram
plt.title('Scaled image histogram multi nivel')
plt.hist(expantion2.ravel(),256,[0,256])
plt.show()

#show the compressed image histogram
plt.title('Compressed image histogram multi nivel')
plt.hist(compresion2.ravel(),256,[0,256])
plt.show()

cv2.waitKey(0)
