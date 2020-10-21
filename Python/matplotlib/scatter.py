# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:02:59 2020
https://www.jianshu.com/p/6027db4a0d2f
散点图

"""

import numpy as np
import matplotlib.pyplot as mp
n = 100
# 172:  期望值
# 10:   标准差
# n:    数字生成数量
x = np.random.normal(172, 5, n) # 均值  标准差  数量
y = np.random.normal(60, 5, n)
mp.figure('scatter', facecolor='lightgray')
mp.title('scatter')

mp.tick_params(labelsize=10)
mp.grid(linestyle=':')

#mp.scatter(x, y, c='red')           #直接设置颜色
d = (x-172)**2 + (y-60)**2       #每个散点 的 L2范数 距离

#散点显示
mp.scatter(x, y, c=d, cmap='jet')    #以c作为参数，取cmap颜色映射表中的颜色值
mp.show()

