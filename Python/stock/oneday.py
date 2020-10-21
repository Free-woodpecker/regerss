# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:06:20 2020

@author: 爬上阁楼的鱼
"""

import baostock as bs
import pandas as pd
import stock as sk
import pymssql


def SaveStock(connect, code, codeName, types = 0):
    cursor = connect.cursor()
    sql = 'insert into [day.' + code + '] (StockId, [date],[open],[high],[low],[close],[preclose],[volume],[amount],adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST) values'
    


def download_data(date):
    bs.login()

    stock_rs = bs.query_all_stock(date)
    stock_df = stock_rs.get_data()
    data_df = pd.DataFrame()
    for code in stock_df["code"]:
        print("Downloading :" + code)
        k_rs = bs.query_history_k_data_plus(code, 
                                            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
                                            date, date)
        
        data_df = data_df.append(k_rs.get_data())
    bs.logout()
    data_df.to_csv("demo_assignDayData.csv", encoding="gbk", index=False)
    print(data_df)




# connect = pymssql.connect(server='.', user='sa',password='521374335',database='Reason')












