# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:10:24 2020

@author: 爬上阁楼的鱼
"""

import numpy as np

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
        
        i=0
        while i<self.batch_size:
            self.h[i] += self.b[-1]
            i += 1
        
        # 激活
        if self.mode == 1:
            self.h = sigmoid(self.h)
        else:
            self.h = np.tanh(self.h)
        
        return self.h 

    # 计算上一次的loss  需要知道下一次算出本层的损失
    def backward(self, loss):
        self.loss = loss
        return np.dot(loss, self.weights.T)
    
    def train(self, input, learn):
        # 激活求导
        # loss * x
        if self.mode == 1:
            temp = self.loss * sigmoid_derivative(self.h)
        else:
            temp = self.loss * tanh_derivative(self.h) 
        
        # temp = self.loss * sigmoid_derivative(self.h)
        
        # learn * x * (loss * d(y))
        adjustWeight = np.dot(input.T, temp)
        adjustB = temp
        
        adjustB = adjustB.T
        
        i=0
        while i<self.selfNum:
            self.b[i] += learn * np.sum(adjustB[i])/self.selfNum
            i+=1
        
        self.weights += learn * adjustWeight;


num = 400
num_l = 3

training_inputs = np.random.rand(num,2) - 1
training_outputs = np.random.rand(num,1)

# 新模式测试
#############################################
n = int(num/2)

# 172:  期望值
# 10:   标准差
# n:    数字生成数量
x1 = np.random.normal(1, 0.1, n)
y1 = np.random.normal(0.1, 0.1, n)

x2 = np.random.normal(0.1, 0.1, n)
y2 = np.random.normal(1, 0.1, n)

# 一维拼接延长
x = np.append(x1, x2)
y = np.append(y1, y2)

# 一维拼接二维
training_inputs = np.stack((x, y)).T
#############################################


# 标定
i=0
while i < num:
    # 新模式测试
    #######################################
    if i < 200:
        training_outputs[i][0] = 1
    else:
        training_outputs[i][0] = 0
    #######################################
    
    # 原始模式
    # if (training_inputs[i][0]**2 + training_inputs[i][1]**2) >1:
    #     training_outputs[i][0] = 1
    # else:
    #     training_outputs[i][0] = 0
    i += 1


layer1 = layer(2,num_l,num)
layer2 = layer(num_l,2,num)
layer3 = layer(2,1,num,1)


# 绘图
i=0
for su in training_inputs:
    if i < 200:
        mp.plot(su[0],su[1], "r.")
    else :
        mp.plot(su[0],su[1], "b.")
    i+=1

# mp.scatter(training_inputs.T[0], training_inputs.T[1], training_outputs*20 + 1, cmap='jet')    #以c作为参数，取cmap颜色映射表中的颜色值
mp.show()


i=0
while i<4000:
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
    layer1.train(training_inputs, 0.003)
    layer2.train(layer1.h, 0.003)
    
    # 输出层训练
    layer3.train(layer2.h, 0.003)
    
    if i%200 == 0:
        print(np.sum(error**2)/num)
    i+=1



        