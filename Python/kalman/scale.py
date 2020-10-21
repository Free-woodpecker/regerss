# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 09:27:10 2020

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
with open('213.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))
    
    for row in reader:
        dT.append(row[1])
        biasDiff.append(row[0])
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
W = np.linalg.inv((X.T.dot(X))+np.exp(-11) * np.eye(M+1)).dot(X.T).dot(a)#introduce regularization

W2 = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(a)

# W3 = []:1.004052201847,0.000088956516,0.000001]

W[0] = 1.004052201847
W[1] = 0.000088956516
W[2] = 0.000001

# W[0] = 0.00298982
# W[1] = 0.0542056
# W[2] = -0.000734694
# # W[3] = 0.0

# x = np.arange(10,80,1)

y_estimate=X.dot(W2)
y_estimate2=X.dot(W)

mp.plot(x,y_estimate,'r',lw=2.0)
mp.plot(x,y_estimate2,'g',lw=2.0)

mp.show()
