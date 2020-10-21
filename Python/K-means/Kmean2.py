# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:03:01 2020

@author: 爬上阁楼的鱼
"""
import numpy as np  
from sklearn.cluster import KMeans  
import matplotlib.pyplot as mp

n = 200

# 172:  期望值
# 10:   标准差
# n:    数字生成数量
x1 = np.random.normal(172, 10, n)
y1 = np.random.normal(60, 10, n)

x2 = np.random.normal(20, 10, n)
y2 = np.random.normal(40, 10, n)

x3 = np.random.normal(60, 10, n)
y3 = np.random.normal(190, 10, n)

# 一维拼接延长
x = np.append(x1, x2)
x = np.append(x, x3)

y = np.append(y1, y2)
y = np.append(y, y3)

# 二维拼接扩展
dot = np.stack((x, y)).T

# 初始化  n_clusters 聚类数目
kmeans=KMeans(n_clusters=3)  

# 运算
kmeans.fit(dot)

# 中心点
print(kmeans.cluster_centers_)

# 将数据按类型生成标签
# print(kmeans.labels_)


mp.plot(dots[0][0],dots[0][1], "go")
mp.plot(dots[1][0],dots[1][1], "go")
mp.plot(dots[2][0],dots[2][1], "go")