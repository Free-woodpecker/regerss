# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:18:40 2020

@author: 爬上阁楼的鱼
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import pywt
import pywt.data

def test_pywt():
    # Load image
    original = pywt.data.camera()
    
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    plt.imshow(original)
    plt.colorbar(shrink=0.8)
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.tight_layout()
    plt.show()


test_pywt()