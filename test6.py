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
histgrey = cv2.calcHist([gray], [0], None, [256], [0, 256])
norm_hist = histgrey / (gray.shape[0] * gray.shape[1])
hist_especific_cumm = norm_hist.cumsum()

#equalize the img
eq_gray = cv2.equalizeHist(gray)
eq_hist = cv2.calcHist([eq_gray], [0], None, [256], [0, 256])
eq_cumulative_distribution_function = eq_hist.cumsum()
#or
"""
eq = np.zeros((image.shape[0],image.shape[1]),dtype = np.uint8)
eq_hist = np.zeros(256)

for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        eq[i,j] = 255 * cumulative_distribution_function[gray[i,j]]
        eq_hist[eq[i,j]] += 1

eq = np.array(eq)
eq_hist = np.array(eq_hist)
"""

# generate a new histogram from an arbritary function
hist_especific = np.zeros(256)
for i in range(0, 255):
    hist_especific[i] = np.cos(i/64) * 128 + 128

# normalize the histogram
hist_especific = hist_especific / (gray.shape[0] * gray.shape[1])
mapped_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
hist_especific_cumm = hist_especific.cumsum()

#map the gray values to the new histogram
for i in range(0, gray.shape[0]):
    for j in range(0, gray.shape[1]):
        index = eq_gray[i, j]
        mapped_img[i, j] = 256 * hist_especific_cumm[index]

mapped_img = np.array(mapped_img)

# generate a histogram of the mapped image
hist_final_img = cv2.calcHist([mapped_img], [0], None, [256], [0, 256])
# plot all the histograms
# cutoff any ferequenc above the mean
fig, ax = plt.subplots()
ax.plot(histgrey, label='Original', alpha=1)
ax.legend()
ax.set(xlabel='Gray value', ylabel='Frequency',
       title='Original')
ax.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(eq_hist, label='Equalized', alpha=1)
ax.set(xlabel='Gray value', ylabel='Frequency',
       title='Equalized')
ax.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(hist_final_img, label='Mapped', alpha=1)
ax.set(xlabel='Gray value', ylabel='Frequency',
       title='Mapped')
ax.grid()
plt.show()

# show the new histogram
fig, ax = plt.subplots()
ax.plot(hist_especific, label='Normalized', alpha=1)
ax.plot(hist_especific_cumm 
        , label='Cumulaive', alpha=1)
ax.legend()
ax.set(xlabel='Gray value', ylabel='Frequency',
       title='Histogram')
ax.grid()
plt.show()


# display all the images

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Equalizided", eq_gray)
cv2.imshow("Mapped", mapped_img)
cv2.waitKey(0)
