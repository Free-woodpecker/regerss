# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:53:07 2020

@author: 爬上阁楼的鱼
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
with open('20602.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))
    
    for row in reader:
        dT.append(row[1])
        biasDiff.append(row[3])
        num += 1
        testx.append(num)

a = np.asarray(biasDiff, dtype =  float)

x = np.asarray(dT, dtype =  float)
# x = np.asarray(testx, dtype =  float)

mp.plot(x, a, lw = 1.0)
# mp.scatter(x, a, lw = 1.0)

X=x
for i in range(2,M+1):
    X = np.column_stack((X, pow(x,i)))


X = np.insert(X,0,[1],1)

# 投影矩阵 (A^t * A)^-1 * A^t
# _w = (A^t * A)^-1 * A^t * y
W = np.linalg.inv((X.T.dot(X))+np.exp(-8) * np.eye(M+1)).dot(X.T).dot(a)#introduce regularization
y_estimate=X.dot(W)

mp.plot(x,y_estimate,'r',lw=2.0)


mp.show()


    
