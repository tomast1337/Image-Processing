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
"""
#or
gray = np.zeros((image.shape[0],image.shape[1]))
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        gray[i][j] = image[i][j].sum() // 3

gray = np.array(gray)
"""

lower_bount = 150
upper_bount = 200

# scale the inner bounds to 0-255

scaled = np.zeros((image.shape[0],image.shape[1]))
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        scaled[i,j] = 255 * (gray[i,j] - lower_bount) // (upper_bount - lower_bount)

scaled = np.array(scaled)

# Show the images
cv2.imshow('Original image',gray)
cv2.waitKey(0)

#show the original image histogram
plt.hist(gray.ravel(),256,[0,256])
plt.show()

cv2.imshow('Scaled image', scaled)
cv2.waitKey(0)

#show the scaled image histogram
plt.hist(scaled.ravel(),256,[0,256])
plt.show()