# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:12:12 2020

@author: 爬上隔离的鱼
"""

import csv
import numpy as np
import matplotlib.pyplot as mp

# M 阶多项式 
M = 2
gyrox = []
gyroy = []
gyroz = []
temp = []
num = 0


test = []

i = 0
with open('111.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))
    
    for row in reader:
        
        temp.append(row[0])
        gyroz.append(row[1])
        
        num += 1
        
        # print(row[2],end=',')
        
        # #测试
        # test.append(num*num + 10)
        # testx.append(num)
        # if num == 7:
        #     break


a = np.asarray(gyroz, dtype =  float)  
x = np.asarray(temp, dtype =  float)*0.01

# a = np.asarray(test, dtype =  float)  
# x = np.asarray(testx, dtype =  float)

mp.plot(x, a)

X=x
for i in range(2,M+1):
    X = np.column_stack((X, pow(x,i)))


X = np.insert(X,0,[1],1)

# 投影矩阵 (A^t * A)^-1 * A^t
# _w = (A^t * A)^-1 * A^t * y
W = np.linalg.inv((X.T.dot(X))+np.exp(-8) * np.eye(M+1)).dot(X.T).dot(a)#introduce regularization
y_estimate=X.dot(W)

mp.plot(x,y_estimate,'r',lw=4.0)


mp.show()


    