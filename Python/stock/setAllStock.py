# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:56:31 2020

@author: 爬上阁楼的鱼
"""

import baostock as bs
import pandas as pd
import stock as sk
import pymssql
import time

date_start = "2020-07-16"
date_end = "2020-07-16"

def GetStock(connect, bs):
    sql = "select * from [Stock]"
    cursor = connect.cursor()
    cursor.execute(sql)

    i=0
    j=0
    bs.login()
    for row in cursor:
        if i >= -1:
            j+=1
            if j==50 :
                bs.logout()
                j=0
                print('休息5s')
                time.sleep(5)
                bs.login()
                print('重新开始')
            
            name = 'day.' + row[1]
            
            # 获取数据 并 保存数据
            df = sk.GetStockInfoDays(bs, row[1],date_end,date_start)
            # 保存数据
            addStockDayInfo(df, row)
        i+=1
    
    bs.logout()
    cursor.close()   # 关闭游标


def addStockDayInfo(df, row):
    connect = pymssql.connect(server='.', user='sa',password='521374335',database='Reason')
    cursor = connect.cursor()
    num = len(df.index)
    
    if num == 0:
        #connect.commit() # 提交
        cursor.close()   # 关闭游标
        connect.close()
        print('接收错误  无数据')
        return
        #print('query_history_k_data_plus respond error_code:'+df.error_code)
        #print('query_history_k_data_plus respond  error_msg:'+df.error_msg)
        raise Exception('数据接收异常')
    
    print(row)
    i=0
    while i<num: 
        sql = 'insert into [day.' + row[1] + '] (StockId, [date],[open],[high],[low],[close],[preclose],[volume],[amount],adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST) values'
        
        ls = df.loc[i].tolist()
        sqlin = str(row[0])
        j = 0
        for rows in ls:
            if j == 0:
                sqlin +=  ",'" + rows + "'"
            elif j == 1:
                j += 1
                continue
            else:
                if rows is '':
                    rows = '0'
                sqlin +=  ',' + rows
            j += 1
        
        sql += ' ( ' + sqlin + ')'
        #print(sql)
        cursor.execute(sql)
        
        i+=1
    
    connect.commit() # 提交
    cursor.close()   # 关闭游标
    connect.close()


# 登录sql
connect = pymssql.connect(server='.', user='sa',password='521374335',database='Reason')
if connect:
    print("连接成功!")
else :
    print('链接失败')
    exit

GetStock(connect, bs)



connect.close()



