# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 08:52:10 2020
@author: 爬上阁楼的鱼
"""

import pymssql

date_start = "2020-07-03"
date_end = date_start 



# 登录sql
connect = pymssql.connect(server='.', user='sa',password='521374335',database='Reason')
connects = pymssql.connect(server='.', user='sa',password='521374335',database='Reason')
if connect:
    print("连接成功!")
else :
    print('链接失败')
    exit

ls = []

sql = "select * from [Stock]"
cursor = connect.cursor()
cursor.execute(sql)

for row in cursor:
    ls.append(row)
    
cursors = connects.cursor()
for row in ls:
    sql = 'select * from [day.' + row[1] + '] where [date] = ' + "'" + date_start + "'"
    cursor.execute(sql)
    print(sql)
    
    for rows in cursor:
        sql = 'delete [day.' + row[1] +'] where Id = ' + str(rows[0])
        # print(sql)
        cursors.execute(sql)
        connects.commit() # 提交
    

cursor.close()   # 关闭游标
cursors.close()   # 关闭游标

connect.close()
connects.close()

