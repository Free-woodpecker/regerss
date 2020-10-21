# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:06:49 2020
https://www.jianshu.com/p/6027db4a0d2f
等高线图

"""
import numpy as np
import matplotlib.pyplot as mp
n = 1000
# 生成网格化坐标矩阵
x, y = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))
# 根据每个网格点坐标，通过某个公式计算z高度坐标
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
mp.figure('Contour', facecolor='white')
mp.title('Contour', fontsize=20)


# #坐标系标注
#mp.xlabel('x', fontsize=14)
#mp.ylabel('y', fontsize=14)

#
#mp.tick_params(labelsize=10)

#坐标系网格
mp.grid(linestyle=':')

# 绘制等高线图
mp.contourf(x, y, z, 8, cmap='jet')
cntr = mp.contour(x, y, z, 8, colors='black',
                  linewidths=0.5)

# 为等高线图添加高度标签
mp.clabel(cntr, inline_spacing=1, fmt='%.1f', fontsize=10)

mp.show()

