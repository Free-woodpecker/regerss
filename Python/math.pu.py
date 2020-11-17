# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:23:07 2020

@author: 爬上阁楼的鱼
"""

import math

print(math.atan(1/13)*57.3)

w = []


w.append(1)
w.append(0)
w.append(0)
w.append(0)

q = 3.1415926/2

w[0] = math.cos(q/2)
w[1] = math.sin(q/2)*0
w[2] = math.cos(q/2)*0
w[3] = math.cos(q/2)*1


print(w)


