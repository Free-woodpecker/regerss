# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:43:59 2020
动态气泡

"""

# 生成动画泡泡
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.animation as ma

# 构建100个泡泡，确定属性
n = 100
balls = np.zeros(100, dtype=[
    ('position', float, 2),
    ('size', float, 1),
    ('growth', float, 1),
    ('color', float, 4)])
# 初始化所有ball的属性值
# 随机生成最小值为0，最大值为1的N行2列的数组
# 初始化所有ball图标
balls['position'] = np.random.uniform(0, 1, (n, 2))  # n x 2 维矩阵
balls['size'] = np.random.uniform(40, 300, n)        # n x 1 维列向量
balls['growth'] = np.random.uniform(10, 20, n)       # n x 1
balls['color'] = np.random.uniform(0, 1, (n, 4))     # n x 4
# 画图
mp.figure('Animation', facecolor = 'lightgray')
mp.title('Animation', fontsize = 16)
mp.xticks([])
mp.yticks([])
#散点显示                      x                       y                   r               rgb
sc = mp.scatter(balls['position'][:, 0], balls['position'][:, 1], balls['size'], color=balls['color'])
def update(number):
    # 定义更新图像
    balls['size'] += balls['growth']
    # 让某个球重新初始化属性
    ind = number % n
    balls[ind]['size'] = np.random.uniform(40, 300, 1)
    balls[ind]['position'] = np.random.uniform(0, 1, (1, 2))
    sc.set_sizes(balls['size'])
    sc.set_offsets(balls['position'])
anim = ma.FuncAnimation(mp.gcf(), update, interval=30)
mp.show()



