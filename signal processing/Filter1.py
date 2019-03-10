#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:12:26 2019

@author: mahparsa
"""

from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(280490)
x = np.random.randn(101).cumsum()
#gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)¶
#Different sigma: standard deviation for Gaussian kernel
y1 = gaussian_filter1d(x, 1)
y2 = gaussian_filter1d(x, 2)
y3 = gaussian_filter1d(x, 3)
y4 = gaussian_filter1d(x, 4)

plt.plot(x, 'k', label='original data')
plt.plot(y1, '--', label='filtered, sigma=1')
plt.plot(y2, ':', label='filtered, sigma=2')
plt.plot(y3, '-', label='filtered, sigma=3')
plt.plot(y4, '.-', label='filtered, sigma=4')

plt.legend()
plt.grid()
plt.show()