# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:45:13 2020
寻找箱体
@author: 爬上阁楼的鱼
"""


from sqlalchemy import create_engine
from Infos import sh000001
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as mp
import box

numlen = 30


engine = create_engine("mssql+pymssql://sa:521374335@localhost/Reason")
session = sessionmaker(engine)
mySession = session()

# 查询数据
result = mySession.query(sh000001).filter(sh000001.date > '2010-05-10').all()
length = len(result)
num = np.zeros(length)

# 填充数据
i=0
for row in result:
    num[i] = row.open
    i += 1

boxs = []

# 数据分块
i=0
slices = box.slice_num(num, numlen) # 获取可分割次数
while i < slices: #length - numlen:
    # 分割数据并存储进列表
    boxs.append(box.Box(num[i:i+numlen],i))
    i += 1


# 首先要有一段好的数据 有了好的数据, 先天、先验的东西才有了结合经验的立足点 
# 有了好的数据, 按照定义寻找可能的区间 
# 寻找到了可能的区间 按照 










# # 绘图观察
# i=0
# z = 0
# for row in result:
#     if i > numlen:
#         # mp.plot(i - numlen,row.pctChg*0.1, 'b.')
#         # mp.plot(i - numlen,row.close*0.001 - 2.9, 'r.')
#         z += row.pctChg
#         mp.plot(i,z, "g.")      # 变化率的积分大小
#     i+=1
    
#     if i > 200 + numlen:
#         break

# 一阶低通滤波器
a = 0.8
out = boxs[0].rateShape  # 设定初始状态

i=0
for su in boxs:
    out = su.rateShape * a + (1 - a) * out
    mp.plot(i,out, "y.")  # /10 - 0.9
    
    i+=1

    if i> 200:
        break

mp.show()












