# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:45:03 2023

@author: Nicolas Vycas Nery
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('super_imgtetris-vicio.webp')

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

# generate a histogram of the grayscale image
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#or
"""
hist = np.zeros(256)
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        hist[gray[i,j]] += 1

hist = np.array(hist)
"""
#normalize the histogram
norm_hist = hist / (gray.shape[0] * gray.shape[1])


# plot the  histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

# plot the normalized histogram
plt.figure()
plt.title("Grayscale Normalized Histogram")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.plot(norm_hist)
plt.xlim([0, 256])
plt.show()

# compute the cumulative distribution function
cdf = norm_hist.cumsum()
# plot the cumulative distribution function
plt.figure()
plt.plot(cdf)
plt.title("Cumulative Distribution Function")
plt.xlabel("Intensity")
plt.ylabel("Cumulative Probability")
plt.xlim([0, 256])
plt.show()

# plot the histogram and cumulative distribution function
fig, ax = plt.subplots()
ax.plot(norm_hist, color="red", label="Histogram")
ax2 = ax.twinx()
ax2.plot(cdf, color="blue", label="Cumulative Distribution Function")
ax.set_xlim([0, 256])
ax.set_xlabel("Intensity")
ax.set_ylabel("Frequency")
ax2.set_ylabel("Cumulative Probability")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()


cv2.imshow('Original image',image)
cv2.imshow('Expanded image', grey)

cv2.waitKey(0)
