# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:07:09 2020
新建所有 Stock 表
@author: 爬上阁楼的鱼

"""

import pymssql
import time

creatTableSql = '''
(
	Id int primary key identity(1,1) not null,
	StockId int not null,
	[date] datetime not null default getdate(),
	[open] float not null,
	[high] float not null,
	[low] float not null,
	[close] float not null,
	preclose decimal not null,
	volume decimal not null,
	amount decimal not null,
	adjustflag int not null,
	turn float not null,
	tradestatus int not null,
	pctChg float not null,
	peTTM float not null,
	psTTM float not null,
	pcfNcfTTM float not null,
	pbMRQ float not null,
	isST int not null
)
'''

def CreateTable(code):
    connects = pymssql.connect(server='.', user='sa',password='521374335',database='Reason')
    cursor = connects.cursor()
    name = '[day.' + str(code) + ']'
    sql = 'create table ' + name + creatTableSql
    cursor.execute(sql)
    connects.commit()
    cursor.close()  # 关闭游标
    connects.close()

def GetStock(connect, cursor):
    sql = "select * from [Stock]"
    cursor.execute(sql)
    #row = cursor.fetchone()
    #while row:
    for row in cursor:
        # print(row[2])
        CreateTable(row[1])
        print(row[1])
        time.sleep(0.01)

connect = pymssql.connect(server='.', user='sa',password='521374335',database='Reason')

cursor = connect.cursor()  # 创建一个游标对象,python里的sql语句都要通过cursor来执行

GetStock(connect, cursor)

# name = '[day.' + 'sz.300635' + ']'
# sql = 'create table ' + name + creatTableSql

#print(sql)

cursor.close()  # 关闭游标
connect.close()  # 关闭连接


