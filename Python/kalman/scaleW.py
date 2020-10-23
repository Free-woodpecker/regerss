# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:46:50 2020

@author: xing
"""


import csv
import numpy as np
import matplotlib.pyplot as mp

# M 阶多项式 
M = 2
biasDiff = []
num = 0

i = 0

x = []
a = []
x1=0.0
x2=0
x3=0
x4=0

y=0
xy = 0
x2y = 0

with open('213.csv', 'r') as f:
    reader = csv.reader(f)
    
    for row in reader:
        x.append(float(row[1]))
        a.append(float(row[0]))
        
        x1 = x1 + float(row[1])
        x2 = x2 + float(row[1])**2
        x3 = x3 + float(row[1])**3
        x4 = x4 + float(row[1])**4
        
        y = y + float(row[0])
        xy = xy + float(row[0]) * float(row[1])
        x2y = x2y + float(row[0]) * float(row[1])**2

        num += 1

x = np.asarray(x, dtype =  float)

# mp.plot(x, a, lw = 1.0)
mp.scatter(x, a, lw = 1.0)

X=x
for i in range(2,M+1):
    X = np.column_stack((X, pow(x,i)))

X = np.insert(X,0,[1],1)

# 投影矩阵 (A^t * A)^-1 * A^t
# _w = (A^t * A)^-1 * A^t * y
# 正则化
# W = np.linalg.inv((X.T.dot(X))+np.exp(-7) * np.eye(M+1)).dot(X.T).dot(a)
# 无正则化
W2 = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(a)

W =  np.zeros(3)
A = np.array([[num,x1,x2],[x1,x2,x3],[x2,x3,x4]])
b = np.array([[y],[xy],[x2y]])

A2 = np.linalg.inv(A)

# W = np.linalg.inv(A).dot(b)
# -0.101493999362,-0.350430995226,0.024311399087
W[0] = 1.047689138137
W[1] = -0.002908761835
W[2] = 0.000045950889

y_estimate=X.dot(W2)
y_estimate2=X.dot(W)


mp.plot(x,y_estimate,'r',lw=1.0)
mp.plot(x,y_estimate2,'g',lw=2.0)

mp.show()

