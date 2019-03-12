#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:13:22 2019

@author: mahparsa
"""
# the sourse code can be found here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
from scipy.ndimage import gaussian_filter
from scipy import misc
import matplotlib.pyplot as plt
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(141)  
ax2 = fig.add_subplot(142)  
ax3 = fig.add_subplot(143)  
ax4 = fig.add_subplot(144)  
ascent = misc.ascent()
#We can get an 8-bit grayscale bit-depth, accent-to-the-top.jpg
#gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
#using gaussian_filter with different values for sigma. 
result1 = gaussian_filter(ascent, sigma=5)
result2 = gaussian_filter(ascent, sigma=10)
result3 = gaussian_filter(ascent, sigma=15)

ax1.imshow(ascent)
ax2.imshow(result1)
ax3.imshow(result2)
ax4.imshow(result3)
plt.show()
