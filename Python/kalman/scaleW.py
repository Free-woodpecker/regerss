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
x5=0
x6 = 0

y=0
xy = 0
x2y = 0
x3y = 0

with open('213.csv', 'r') as f:
    reader = csv.reader(f)
    
    for row in reader:
        x.append(float(row[1]))
        a.append(float(row[0]))
        
        x1 = x1 + float(row[1])
        x2 = x2 + float(row[1])**2
        x3 = x3 + float(row[1])**3
        x4 = x4 + float(row[1])**4
        x5 = x5 + float(row[1])**5
        x6 = x6 + float(row[1])**6
        
        y = y + float(row[0])
        xy = xy + float(row[0]) * float(row[1])
        x2y = x2y + float(row[0]) * float(row[1])**2
        x3y = x3y + float(row[0]) * float(row[1])**3
        
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

# 三阶拟合  微积分方法CESHI
W =  np.zeros(4)
A = np.array([[num,x1,x2,x3],[x1,x2,x3,x4],[x2,x3,x4,x5],[x3,x4,x5,x6]])
b = np.array([[y],[xy],[x2y],[x3y]])

# A2 = np.linalg.inv(A)

W = np.linalg.inv(A).dot(b)

# W[0] = 1.003301327017
# W[1] = -0.000070400192
# W[2] = -0.000000703356
# W[3] = 0

y_estimate=X.dot(W2)
y_estimate2= W[0] + W[1]*x +W[2]*x**2 + W[3]*x**3


mp.plot(x,y_estimate,'r',lw=1.0)
mp.plot(x,y_estimate2,'g',lw=2.0)

mp.show()

