# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:34:49 2020

@author: 爬上阁楼的鱼
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1,100,100)  # 1 ~ 100

# y =  np.sin(x*3.1415926)

y = np.random.normal(0, 1, size=(x.shape[0])) + 0.05*x
y2 = []
y3 = []


plt.plot(x,y,'r')


x1 = 0
x2 = 0

Ts = 0.015
T = 0.6

for i in range(0,x.shape[0]):
    x1_t = x1
    x1 = x1 + Ts*x2
    x2 = x2 - Ts*(1/T**2*(x1_t - y[i]) + 2/T * x2)
    y2.append(x2)
    y3.append(y2[i] + x2)

plt.plot(x,y2,'b')
plt.plot(x,y3,'g')