# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:28:45 2020

@author: 爬上阁楼的鱼
"""

import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft


gyrox = []
gyroy = []
gyroz = []
temp = []
num = 0


# test = []
testx = []

i = 0
with open('345.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))
    
    for row in reader:
        gyrox.append(row[0])
        gyroy.append(row[1])
        gyroz.append(row[2])
        temp.append(row[6])
        num += 1
        
        print(row[2],end=',')
        
        # #测试
        # test.append(num*num + 10)
        testx.append(num)
        # if num == 7:
        #     break
        
        # mp.plot(row[6],row[0], "r.")
        # mp.plot(row[1],row[6], "b.")
        # mp.plot(row[2],row[6], "y.")

a = np.asarray(gyroz, dtype =  float)  
# x = np.asarray(temp, dtype =  float)
x = np.asarray(testx, dtype =  float)


fft_y=fft(a)


abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
angle_y=np.angle(fft_y)              #取复数的角度



plt.figure()
plt.plot(x,abs_y)   
plt.title('双边振幅谱（未归一化）')
 
plt.figure()
plt.plot(x,angle_y)   
plt.title('双边相位谱（未归一化）')
plt.show()













