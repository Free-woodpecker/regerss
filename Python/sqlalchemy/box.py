# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:29:29 2020

@author: 爬上阁楼的鱼
"""
from sqlalchemy import create_engine
from Infos import sh000001
from sqlalchemy.orm import sessionmaker
import numpy as np
# import matplotlib.pyplot as mp


# numlen = 30



class Box:
    # 基本数据
    max = 0
    min = 0
    mean = 0        # 求分布特性需要
    var = 0
    length = 0
    
    amplitude = 0
    rateShape = 0
    
    # 回归数据
    a = np.zeros(3)   # 二阶最小二乘估计(a0,a1,a2) --> 判断凹凸性 曲率
    
    # IOU 计算需要知道对角线上的点
    # Box的坐标定位
    tl = np.zeros(2)   # 左上点
    br = np.zeros(2)   # 右下点
    
    width = 0
    height = 0
    
    # 先天划定研究的长度范围(经验取决于历史文章  先天取决于人的理解)
    
    
    # 构造函数初始化数据  ls --> Box 的一维数据   dotStart --> Box 的起点x坐标
    def __init__(self, ls, dotStart):
        # 基本
        self.max = np.max(ls)
        self.min = np.min(ls)
        self.mean = np.mean(ls)
        self.var = np.std(ls)
        self.length = len(ls)
        
        self.amplitude = self.max - self.min
        self.rateShape = self.amplitude / self.length
        
        # 区域箱体坐标
        
        self.tl = [dotStart , np.max(ls)]
        self.br = [dotStart + self.length, np.min(ls)]
        
        self.width = self.length
        self.height = self.max - self.min
        
        # 回归
        x = np.arange(self.length)
        self.a = np.polyfit(x, ls, 2)
    
    def area(self):
        return (self.br[0] - self.tl[0])*(self.tl[1] - self.br[1])



# 此处的 IOU 计算注意细节
# 1. 需要判断是否有相交部分！ 有相交部分再进行计算                                        ----
# 2. 思考 IOU 保存在哪里？  IOU 是两个区域之间的特性, 而非一个区域的特性                   ----
# 3. IOU 只反映了量的关系, 没有反映出 幅度之间公共部分 与 时间之上的公共部分(但是时间上的
#     公共部分应当认为是已知的信息 --> 时间轴是人为划定的)                                ----
# 4. 
# 


class IOU:
    iou = -1
    isIntersection = False
    
    def __init__(self, box1, box2):
        # 问题的转化  观察IOU  同样的事物可以有那些不同的定义 !!!  定义 范畴
        in_h = min(box1.tl[1], box2.tl[1]) - max(box1.br[1], box2.br[1])
        in_w = min(box1.br[0], box2.br[0]) - max(box1.tl[0], box2.tl[0])
        
        if in_h>0 and in_w>0:
            intersection = in_h * in_w
            union = box1.area() + box2.area() - intersection
            self.iou = intersection / union
            self.isIntersection = True
        

# 滑动平均
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# 计算按 固定 长度切割数据最大切割次数
def slice_num(a, length):
    ls = len(a)
    return ls//length

# 按固定长度length切割数据 n --> 切第几次(0开始)
def slice_data(a, length, n):
    return a[length*n : length*(n+1)]



# # 登录数据库
# engine = create_engine("mssql+pymssql://sa:521374335@localhost/Reason")
# session = sessionmaker(engine)
# mySession = session()

# # 按 class 查询数据  限定时间条件
# result = mySession.query(sh000001).filter(sh000001.date > '2019-05-10').all()

# length = len(result)


# num = np.zeros(length)

# # 填充数据
# i=0
# for row in result:
#     num[i] = row.open
#     i += 1
#     # print(row.Num)

# 最大最小值
# print('Max:', np.max(num))
# print('Min:', np.min(num))

# 滑动均值曲线
#x = moving_average(num,10)

# boxs = []
# ious = []

# # 数据分块
# i=0
# slices = slice_num(num, numlen) # 获取可分割次数
# while i < length - numlen:
#     # 分割数据并存储进列表
#     boxs.append(Box(num[i:i+numlen],i))
#     i += 1

# i=0
# while i< length - numlen - 1:
#     ious.append(IOU(boxs[i],boxs[i+1]))
#     i += 1

# # 绘制图像
# i=0
# for row in result:
#     if i > numlen:
#         mp.plot(i - numlen,row.pctChg*0.1, 'b.')
#         mp.plot(i - numlen,row.close*0.001 - 2.9, 'r.')
    
#     i+=1
    
#     if i> 400 + numlen:
#         break

# # 一阶低通滤波器
# a = 0.1
# out = ious[0].iou # 设定初始状态

# i=0
# for su in ious:
#     out = su.iou * a + (1 - a) * out
#     mp.plot(i,out - 0.9, "y.")
#     i+=1
    
#     if i> 400:
#         break

# mp.show()








