# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:42:10 2020
极坐标

"""


import numpy as np
import matplotlib.pyplot as mp
mp.figure('Polar', facecolor='orangered')
mp.gca(projection='polar')
mp.title('Polar')
mp.xlabel(r'$\theta$', fontsize=14)
mp.xlabel(r'$\rho$', fontsize=14)
mp.grid(linestyle=':')
# 绘制线性关系
# t = np.linspace(0, 4*np.pi, 1000)
# r = 0.8*t
# mp.plot(t, r)
# mp.show()
# 绘制sin曲线
x = np.linspace(0, 6*np.pi, 1000)
y = 3*np.sin(6*x)
mp.plot(x, y)
mp.show()

