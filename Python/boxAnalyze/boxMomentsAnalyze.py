# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:16:54 2020
区域矩分析箱体区间
@author: 爬上阁楼的鱼
"""

import cv2 as cv
import numpy as np



M = cv.moments(np.random.rand(5,2))

# print(np.sum(contour))  # m00
# print(np.sum(contour[...,0]))  # m00

# 第一行的二阶矩 --> mu02

