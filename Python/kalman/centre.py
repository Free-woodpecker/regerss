# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:35:16 2020

@author: xing
"""

import csv
import numpy as np
import matplotlib.pyplot as mp

# M 阶多项式 
M = 2
biasDiff = []
num = 0

dT = []
testx = []

i = 0

firstT=0
firstS=0

with open('1232.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))
    
    for row in reader:
        firstT = float(row[2])
        firstS = float(row[0])
        break
    
    for row in reader:
        dT.append(float(row[2]) - firstT)
        # dT.append(float(row[2]) - Tlast)
        # Tlast = float(row[2])
        biasDiff.append(float(row[0]) - firstS)
        num += 1
        testx.append(num)

a = np.asarray(biasDiff, dtype =  float)

x = np.asarray(dT, dtype =  float)

mp.scatter(x, a, lw = 1.0)

zfz = 0
fz = 0
z = 0

for i in range(0, 309):
    zfz = zfz + x[i]*a[i]
    fz = fz + a[i]
    z = z + x[i]

print(zfz/fz)
print(zfz/z)

# mp.scatter(x, a, lw = 1.0)

mp.show()






    





