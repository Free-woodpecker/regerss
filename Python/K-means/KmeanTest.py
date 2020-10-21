# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:50:45 2020

@author: 爬上阁楼的鱼
"""
import numpy as np
import matplotlib.pyplot as mp

n = 200

# 172:  期望值
# 10:   标准差
# n:    数字生成数量
x1 = np.random.normal(172, 2, n)
y1 = np.random.normal(60, 2, n)

x2 = np.random.normal(20, 2, n)
y2 = np.random.normal(40, 2, n)

x3 = np.random.normal(60, 2, n)
y3 = np.random.normal(190, 2, n)

# 一维拼接延长
x = np.append(x1, x2)
x = np.append(x, x3)

y = np.append(y1, y2)
y = np.append(y, y3)

# 二维拼接扩展
dot = np.stack((x, y)).T

# 大致实现思想
def Kmean(data, dot = np.random.rand(3,2)*100):
    # print('dot : ',dot)
    
    i=0
    
    # K = [[]]
    # K.append()
    # K.append()
    # K.append()
    
    k1 = []
    k2 = []
    k3 = []
    while i<len(data):
        L2_1 = (data[i][0] - dot[0][0])**2 + (data[i][1] - dot[0][1])**2
        L2_2 = (data[i][0] - dot[1][0])**2 + (data[i][1] - dot[1][1])**2
        L2_3 = (data[i][0] - dot[2][0])**2 + (data[i][1] - dot[2][1])**2
        
        
        
        
        
        # 修改 
        if min(L2_1,L2_2,L2_3) == L2_1:
            # K[0].append([data[i][0],data[i][1]])
            k1.append(np.array([data[i][0],data[i][1]]))
        elif min(L2_1,L2_2,L2_3) == L2_2:
            # K[1].append([data[i][0],data[i][1]])
            k2.append(np.array([data[i][0],data[i][1]]))
        elif min(L2_1,L2_2,L2_3) == L2_3:
            # K[2].append([data[i][0],data[i][1]])
            k3.append(np.array([data[i][0],data[i][1]]))
        
        i += 1
    
    # 更新区域中心
    x = 0
    y = 0
    for su in k1:
        x += su[0]
        y += su[1]
    dot[0] = [x/len(k1), y/len(k1)]
    
    x = 0
    y = 0
    for su in k2:
        x += su[0]
        y += su[1]
    dot[1] = [x/len(k2), y/len(k2)]
    
    x = 0
    y = 0
    for su in k3:
        x += su[0]
        y += su[1]
    dot[2] = [x/len(k3), y/len(k3)]
    
    # return dot, K
    return dot, k1, k2, k3

# 首次随机生成种子
dots, k1, k2, k3 = Kmean(dot)
# dots, K = Kmean(dot)

# 迭代搜索
i=0
while i<40:
    dots, k1, k2, k3 = Kmean(dot,dots)
    # dots, K = Kmean(dot,dots)
    # print(dots)
    i+=1


print(dots)


# 绘图
for su in k1:
    mp.plot(su[0],su[1], "r.")

for su in k2:
    mp.plot(su[0],su[1], "b.")

for su in k3:
    mp.plot(su[0],su[1], "y.")

# 绘制中心点
mp.plot(dots[0][0],dots[0][1], "go")
mp.plot(dots[1][0],dots[1][1], "go")
mp.plot(dots[2][0],dots[2][1], "go")

mp.show()


# print()

# 打印聚类中心
# print(dots)
