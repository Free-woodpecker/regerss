# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:36:07 2020

@author: 爬上阁楼的鱼
"""


import numpy  
import pylab
 
def KalmanFilter(z,  n_iter = 20):  
    #这里是假设A=1，H=1的情况  
      
    # intial parameters  
     
    sz = (n_iter,) # size of array   
    
    
    #分析 
    # Q R 两个参数 如果影响效果
    
    #Q = 1e-5 # process variance  
    # 过程方差
    Q = 0.00003 # process variance   

    xhat=numpy.zeros(sz)      # a posteri estimate of x  x的后验估计
    P=numpy.zeros(sz)         # a posteri error estimate 后验误差估计
    xhatminus=numpy.zeros(sz) # a priori estimate of x   x的先验估计
    Pminus=numpy.zeros(sz)    # a priori error estimate  先验误差估计
    K=numpy.zeros(sz)         # gain or blending factor  增益或混合系数
    
    # 测量方差的估计 --> R 小了显著提升系统的响应速度
    R = 0.5**2 # estimate of measurement variance, change to see effect  
    
    # intial guesses
    xhat[0] = z[0]
    P[0] = 1.0
    A = 1
    H = 1
    
    # Q 的大小影响先验误差的大小值 (正比)
    # R 的大小影响增益的大小 (正比)
    # 增益的大小影响后验误差大小
    
    # https://blog.csdn.net/dan1900/article/details/41206449
    # Q 估计的噪声(covariance)  --> Q 越大  实际更偏向测量
    # R 测量的噪声(covariance)  --> R 越大  实际更偏向于估计
    
    for k in range(1,n_iter):
        # 动态修改
        # if k == 220:
        #     Q = 0.01
        
        # time update
        xhatminus[k] = A * xhat[k-1]                            #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0   x的先验估计
        Pminus[k] = A * P[k-1] + Q                              #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1 更新先验误差
        
        # measurement update
        K[k] = Pminus[k]/(Pminus[k] + R)                             #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1 更新增益或混合系数
        xhat[k] = xhatminus[k] + K[k] * (z[k] - H * xhatminus[k])    #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1  
        P[k] = (1-K[k] * H) * Pminus[k]                              #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
        # k 增益系数更新系统的后验误差  认定这一时刻与下一时刻相同  所以 A = 1 如果系统随时间而改变则此项可能需要变化.
        # 系统没有控制量  所以  U(K) = 0
    
    return xhat, P


# 对比测试
def FIR(z,  n_iter=20):
    sz = (n_iter,)
    
    xhat=numpy.zeros(sz)
    xhat[0] = z[0]
    
    alpha = 0.985
    for k in range(1, n_iter):
        xhat[k] = xhat[k-1]*alpha + z[k-1]*(1 - alpha)
    
    return xhat


if __name__ == '__main__':
    # 文件形式读取数据
    # with open("raw_data.txt", "r", encoding="utf-8") as f:
    #     text = f.readline().split(",")
    
    raw_data = list()
    
    x1 = numpy.random.normal(20,5,200)
    x2 = numpy.random.normal(0,5,200)
    
    # 拼接数据
    text = numpy.append(x1, x2)  #numpy.random.normal(0,20,100)
    # text = x1
    
    print(numpy.var(text))       # 打印方差
    
    for x in text:
        raw_data.append(int(x))
    
    # KF test
    xhat, P = KalmanFilter(raw_data, n_iter=len(raw_data))
    # 低通 Test
    xhat2 = FIR(raw_data, n_iter=len(raw_data))
    
    # 信息展示
    pylab.plot(raw_data, 'k-', label='raw measurement')  # 原始
    pylab.plot(xhat, 'r-', label='Kalman estimate')      # kalman
    pylab.plot(xhat2, 'y-', label='FIR')                 # FIR
    
    # 坐标轴
    pylab.legend()
    pylab.xlabel('Iteration')
    pylab.ylabel('Data')
    
    # 显示
    pylab.show()
    
    # x的先验估计:上一时刻的状态与约定的过程方差之积
    # 先验误差包含有上一次的后验误差估计, 与约定的过程方差,测量方差有关
    # 增益系数中包含有 先验误差 和 先天设定的 测量方差的估计值(小--> 增益小)
    # x的后验估计 当前x的先验估计 与 增益系数与(当前实际值与先验值只差)之积
    # 后验误差估计 先验误差与增益的背离面

# 看的太简单
#1. 没有相关知识，不存在认识了解，不存在经验
#2. 思维方式问题(面对未知或者在极少了解的情况的态度 --> 模糊, 应当深入了解, 基本信息, 再逆向思考每一步的充分条件)
