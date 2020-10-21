# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:38:42 2020
将所有股票与指数写入数据库
@author: gyf_2
"""

import pymssql
import baostock as bs
import pandas as pd


# 写入所有Stock
def SetAllStock(connect, df):
    cursor = connect.cursor()
    i=0
    num = len(df.index)
    while i< num:
        SaveStock(cursor, result.loc[i,'code'], result.loc[i,'code_name'])
        print(result.loc[i,'code_name'])
        i+=1
    
    connect.commit() # 提交
    cursor.close()   # 关闭游标

# 写入一个 事务方式 未提交
def SaveStock(cursor, code, codeName, types = 0):
    cursor.executemany(
        "insert into Stock (Code, Name, Type) values (%s,%s,%d)",
        [
            (code,codeName,types)
        ]
    )


#### 登陆Sql ####
connect = pymssql.connect(server='.', user='sa',password='521374335',database='Reason')

if connect:
    print("连接成功!")
else :
    print('链接失败')
    exit

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)


#### 获取证券信息 ####
rs = bs.query_all_stock(day="2020-06-19")
print('query_all_stock respond error_code:'+rs.error_code)
print('query_all_stock respond  error_msg:'+rs.error_msg)


#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())


result = pd.DataFrame(data_list, columns=rs.fields)

#### 结果集输出到csv文件 ####   
# result.to_csv("all_stock.csv", encoding="gbk", index=False)

print(len(result.index))

# 写入数据库
SetAllStock(connect, result)

#str = result.loc[3,'code']
#print(str)

#### 登出系统 ####
bs.logout()
connect.close()

