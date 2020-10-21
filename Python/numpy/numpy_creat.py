# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:38:37 2020

@author: 爬上阁楼的鱼
"""
import numpy as np

############ 生成 ############

# 生成 长度为 4 的 向量 默认 float64
x = np.zeros(4)
#print(x[0])  #数

y = np.zeros((4,2))
#print(y[0])  #向量


# 2x2个 (x,y) 坐标
z = np.zeros((2,2),dtype = [('x', 'i4'),('y','i1')])
#print(z['y']) #仅打印出 y 的内容
z['y'][0][0] = 1
z['y'][1][1] = 255

z['x'][0][0] = -1
#print(z)  #打印出 2x2 个坐标


# 2x2 矩阵 
x = np.ones([2,2])
x[0][1] = 2
x[1][0] = 3
x[1][1] = 4

# 打印行向量
#print(x[0])
#print(x[1])


# 创建顺序数组  默认 int32
x = np.arange(5)
print (x)

# 设置了 dtype
x = np.arange(5, dtype =  float)
print (x)

# 设置了 起始值,终止值,步长
x = np.arange(10,20,1)  
print (x)

# 生成等差数列 起始值,终止值,长度
a = np.linspace(1,10,20)
print(a)

# 生成固定值数列 长度 10
a = np.linspace(1,1,10)
print(a)

# 生成数列 不包含终止值
a = np.linspace(10, 20, 5, endpoint =  False)  
print(a)

# 附带间距的等差数列
a = np.linspace(1,10,100,retstep= True)
print(a[1]) #打印间距

# 设定特定格式的 等差数列
b =np.linspace(1,10,20).reshape([4,5])
print(b)



# 生成等比数列 默认底数是10
# 起始 10^1.0 ~ 终止 10^4.0 长度 10
a = np.logspace(1.0,  4.0, num =  10)  
print (a)

# 设定底数为2 起始2^0 ~ 终止2^9 长度100 
a = np.logspace(0,9,100,base=2)
print (a)


############ 现有创建 ############

# 从元胞创建 向量
x =  (1,2,3) 
a = np.asarray(x)  
# print (a)

# 从列表创建 向量
x =  [1,2,3] 
a = np.asarray(x)  
# print (a)


# 使用 range 函数创建顺序列表对象  
list=range(5)
it=iter(list)
 
# 使用迭代器创建 ndarray 
x=np.fromiter(it, dtype=float)
#print(x)




