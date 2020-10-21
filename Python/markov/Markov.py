# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:15:41 2020

@author: 爬上阁楼的鱼
"""
import numpy as np

# 马尔可夫链收敛条件
# 1.可能的状态数是有限的
# 2.状态间的转移概率需要固定不变
# 3.从任意状态能够转变到任意状态
# 4.不能是简单的循环，例如全是从x到y再从y到x

def markov():
    init_array = np.array([0.1, 0.7, 0.2])              # 初始状态向量
    transfer_matrix = np.array([[0.9, 0.075, 0.025],    # 状态转移矩阵 明确3阶
                               [0.15, 0.8, 0.05],
                               [0.25, 0.25, 0.5]])
    
    # transfer_matrix = np.array([[0.1, 0.9, 0],
    #                             [0,   0,   1],
    #                             [0.7, 0,  0.3]])
    
    restmp = init_array
    
    for i in range(300):
        res = np.dot(restmp, transfer_matrix)
        if i%20 == 0:
            print(i, "\t", res)
        restmp = res


def matrixpower():
    transfer_matrix = np.array([[0.9, 0.075, 0.025],
                               [0.15, 0.8, 0.05],
                               [0.25, 0.25, 0.5]])
    restmp = transfer_matrix
    for i in range(25):
        res = np.dot(restmp, transfer_matrix)
        print(i)
        print(res)
        print()
        restmp = res

markov()

# matrixpower()