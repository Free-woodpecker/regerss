# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:44:11 2020

@author: 爬上各路的鱼
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

with open('1231.csv', 'r') as f:
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
# x = np.asarray(testx, dtype =  float)

# mp.plot(x, a, lw = 1.0)
mp.scatter(x, a, lw = 1.0)

X=x
for i in range(2,M+1):
    X = np.column_stack((X, pow(x,i)))

X = np.insert(X,0,[1],1)

# 投影矩阵 (A^t * A)^-1 * A^t
# _w = (A^t * A)^-1 * A^t * y
# 正则化
W = np.linalg.inv((X.T.dot(X))+np.exp(-7) * np.eye(M+1)).dot(X.T).dot(a)
# 无正则化
W2 = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(a)

# W[0] = 0.00298982
# W[1] = 0.0542056
# W[2] = 0
# # W[3] = 0.0

# x = np.arange(10,80,1)

y_estimate=X.dot(W)



mp.plot(x,y_estimate,'r',lw=2.0)


mp.show()

