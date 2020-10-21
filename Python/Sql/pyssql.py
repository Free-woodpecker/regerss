# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""

# import cv2
import pymssql #引入pymssql模块
# import tushare as ts 
import pandas as pd

#import PIL
#import numpy as np

# img = cv2.imread('D:\3.jpg')

# 测试

def SelectTable():
    sql = "SELECT * FROM [Industry]"
    cursor.execute(sql)
    #row = cursor.fetchone()
    #while row:
    for row in cursor:
        print(row)
        #row = cursor.fetchone()
    # 也可以使用for循环来迭代查询结果
    # for row in cursor:
    #     print("ID=%d, Name=%s" % (row[0], row[1]))

print('Start')

#以下两种方法均可
# 注意   设定 管理里面  IPALL
connect = pymssql.connect(server='192.168.8.111', user='sa',password='ysaw,521374335/*',database='Reason')
#connect = pymssql.connect(server='.',user='sa',password='521374335',database='Test')

if connect:
    print("连接成功!")
else :
    print('链接失败')

cursor = connect.cursor()  # 创建一个游标对象,python里的sql语句都要通过cursor来执行



# sql = 'select * from [Jobs]'

# cursor.execute(sql)   #执行sql语句
# #row = cursor.fetchone()  #读取查询结果,
# row = cursor.fetchall()
# print(pd.DataFrame(list(row)).shape)
#connect.commit()  # 提交


#SelectTable()


data = '''
create table SZ50(
    Id int primary key identity(1,1) not null,
    StockId int not null
)
'''
# cursor.execute(data)
# connect.commit()

cursor.close()  # 关闭游标
connect.close()  # 关闭连接






