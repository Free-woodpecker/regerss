# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:10:24 2020

@author: 爬上阁楼的鱼
"""

import numpy as np
import csv
# import math as mh
import matplotlib.pyplot as mp
mp.figure('scatter', facecolor='lightgray')
mp.title('scatter')

mp.tick_params(labelsize=10)
mp.grid(linestyle=':')


def sigmoid(x):
    # 定义 sigmoid 函数
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # 计算Sigmoid函数的导数（用于反向传播）
    return x * (1 - x)

def tanh_derivative(x):
    return 1-np.tanh(x)**2

def ReLU(x):
    return 1 * (x > 0) * x

def ReLU_derivative(x):
    b = x > 0
    b = 1 * b
    return b

def LReLU(x):
    b = x > 0
    b = 1 * b
    d = x <= 0
    d = -0.1 *d
    return x*(b - d)  # ???????

def LReLU_derivative(x):
    b = x > 0
    b = 1 * b
    d = x <= 0
    d = -0.1 * d
    return b - d      # ???????

# 明确 概念 定义, 此乃研究一个问题的基本要义
# 概念存在模糊, 总会存在寸步难行的情况
# 当然可以在实践中学习概念与定义, 但必须是可以获得概念与定义的情况才行.
# 设定原则：
# 1. 优先明确概念定义, 明确知识的范围, 知识的边界在哪里. 在明确概念定义的基础下前进.
# 2. 判定是否明确定义： 尝试自己举例子, 能举出几个栗子则说明对其有一定的了解
    

# 存在问题：
# 反向传播与训练时做的事情划分模糊
# 梯度应当在 反向传播时 计算 不应该堆在训练里面
# 汝的反向传播仅仅计算了 上一层损失大小
class layer:
    def __init__(self, upper, selfNum, batch_size, mode = 0):
        self.weights = 2 * np.random.random((upper, selfNum)) - 1
        self.b = 2 * np.random.random((selfNum, 1)) - 1
        self.selfNum = selfNum
        self.batch_size = batch_size
        self.h = np.random.random((batch_size,selfNum))
        self.loss = np.random.random((batch_size,selfNum))
        self.mode = mode
    
    def forward(self, input):
        input = input.astype(float)
        self.h = np.dot(input, self.weights)
        
        i = 0
        while i < self.batch_size:
            self.h[i] += self.b[-1]
            i += 1
        
        # 激活
        if self.mode == 3:
            self.h = sigmoid(self.h)
        elif self.mode == 2:
            self.h = np.tanh(self.h)
        elif self.mode == 1:
            self.h = ReLU(self.h)
        elif self.mode == 0:
            self.h = LReLU(self.h)
        
        return self.h
    
    # 计算上一次的loss  需要知道下一次算出本层的损失
    def backward(self, loss):
        self.loss = loss
        return np.dot(loss, self.weights.T)
    
    def train(self, input, learn):
        # 激活求导
        # loss * x
        if self.mode == 3:
            temp = self.loss * sigmoid_derivative(self.h)
        elif self.mode == 2:
            temp = self.loss * tanh_derivative(self.h)
        elif self.mode == 1:
            temp = self.loss * ReLU_derivative(self.h)
        elif self.mode == 0:
            temp = self.loss * LReLU_derivative(self.h)
        
        # learn * x * (loss * d(y))
        adjustWeight = np.dot(input.T, temp)
        adjustB = temp
        
        adjustB = adjustB.T
        
        i = 0
        while i<self.selfNum:
            self.b[i] += learn * np.sum(adjustB[i])/self.selfNum
            i += 1
        
        self.weights += learn * adjustWeight;

num_l = 7



biasDiff = []
num = 0

T = []
dT = []
bias = []
testx = []

i = 0
with open('20602.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))
    
    for row in reader:
        T.append(row[0])
        dT.append(row[1])
        bias.append(row[2])
        biasDiff.append(row[3])
        num += 1
        testx.append(num)


training_inputs = np.random.rand(num,3) - 1
training_outputs = np.random.rand(num,1)



#############################################

# 标定
i=0
while i < num:
    training_inputs[i][0] = T[i]
    training_inputs[i][1] = dT[i]
    training_inputs[i][2] = bias[i]
    training_outputs[i][0] = biasDiff[i]
    i += 1

# 网络定义
layer1 = layer(3,num_l,num, 2)
layer2 = layer(num_l,3,num,2)
layer3 = layer(3,1,num,2)

# 绘图
i=0
for su in training_inputs:
    if i < 200:
        mp.plot(su[0],su[1], "r.")
    else :
        mp.plot(su[0],su[1], "b.")
    i+=1

mp.show()

i=0
while i< 18000:
    # 隐含层
    h1 = layer1.forward(training_inputs)
    h2 = layer2.forward(h1)
    # 输出层
    output = layer3.forward(h2)
    
    error = training_outputs - output
    
    l3 = layer3.backward(error)
    l2 = layer2.backward(l3)
    l1 = layer1.backward(l2)
    
    # 隐含层训练
    layer1.train(training_inputs, 0.0002)
    layer2.train(layer1.h, 0.001)
    
    # 输出层训练
    layer3.train(layer2.h, 0.0002)
    
    # print(layer1.h)
    # print(h1)
    # print(layer1.weights)
    # print(np.sum(error**2)/num)
    
    if i%200 == 0:
        print(np.sum(error**2)/num)
    
    # print()
    i+=1



        