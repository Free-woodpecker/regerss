# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:14:12 2020

@author: 爬上阁楼的鱼
"""

import math

Q = [1,0,0,0]

a = 2.4/180*3.14

Q[0] = math.cos(a/2)
# Q[1] = 

Q[3] = 1 * math.sin(a/2)  # z轴旋转

# print(Q)


w = 5270

k = (w/5248 - w/5257)/(47-26)
print(k)


