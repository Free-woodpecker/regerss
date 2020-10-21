# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:22:06 2020
卡尔曼滤波

https://zhuanlan.zhihu.com/p/77327349
@author: 司南牧
"""


import numpy as np
import matplotlib.pyplot as plt

# 模拟数据
t = np.linspace(1,100,100)  # 1 ~ 100

# 加速度
a = 0.5

# s = 1/2*a t^2 零速度 下的位置预测
# 卡尔曼为线性预测, 这类似乎跟踪不好, 需要换 EKF
position = (a * t**2)/2

# 叠加噪声  包含噪声的位置数据
position_noise = position + np.random.normal(0, 120, size=(t.shape[0]))

plt.plot(t,position,label='truth position')
plt.plot(t,position_noise,label='only use measured position')

# 初始状态
# 初试的估计导弹的位置就直接用GPS测量的位置
predicts = [position_noise[0]]
position_predict = predicts[0]

predict_var = 40  # 预测方差  -->  Q   # 初始
odo_var = 120**2  # 位置方差  --> R1
v_std = 30        # 速度方差  --> R2

for i in range(1,t.shape[0]):
    
    # 速度变化率 叠加噪声
    dv =  (position[i] - position[i-1]) + np.random.normal(0,30)  # 模拟从IMU读取出的速度    并叠加噪声
    
    # s1 = s0 + vt 此处界定
    # 先验位置预测  观测噪声速度积分
    position_predict = position_predict + dv                      # 利用上个时刻的位置和速度预测当前位置
    
    # 预测数据来源于 观测到的速度 与 系统变化自身存在的误差
    predict_var += v_std**2                                       # 更新预测数据的方差 
    
    # 下面是Kalman滤波=  先验估计位置(使用观测速度积分)  *  位置方差  /  (先验误差 + 位置方差)   +   观测位置信息叠加噪声 * 先验误差 (预测方差) / (先验误差 + 位置方差)
    position_predict = position_predict * odo_var / (predict_var + odo_var) + position_noise[i] * predict_var / (predict_var + odo_var)
    # 后验误差   =   先验误差 * 位置观测方差 / (先验误差 + 位置观测方差)^2
    predict_var = (predict_var * odo_var) / (predict_var + odo_var)**2
    
    # 输出
    predicts.append(position_predict)



plt.plot(t,predicts,label='kalman filtered position')

plt.legend()
plt.show()