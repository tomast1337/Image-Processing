# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:45:03 2023

@author: Nicolas Vycas Nery
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('img1.webp')

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

# generate a new histogram from an arbritary function
hist2 = np.zeros(256)
for i in range(0,255):
    hist2[i] = (i-128)**2

# normalize the histogram
hist2 = hist2 / (gray.shape[0] * gray.shape[1])

mapped_img = np.zeros((image.shape[0],image.shape[1]),dtype = np.uint8)

cumulative_distribution_function = hist2.cumsum()
cumulative_distribution_function = cumulative_distribution_function * 255 / cumulative_distribution_function[-1]
cumulative_distribution_function = cumulative_distribution_function.astype(np.uint8)


#map the gray values to the new histogram
for i in range(0,gray.shape[0]):
    for j in range(0,gray.shape[1]):
        mapped_img[i,j] = cumulative_distribution_function[gray[i,j]]

mapped_img = np.array(mapped_img)

# generate a histogram of the mapped image
hist3 = cv2.calcHist([mapped_img], [0], None, [256], [0, 256])

# plot the histograms
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

plt.figure()
plt.title("Arbritary Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist2)
plt.xlim([0, 256])
plt.show()

plt.figure()
plt.title("Mapped Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist3)
plt.xlim([0, 256])
plt.show()

# display all the images

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Mapped", mapped_img)
cv2.waitKey(0)
