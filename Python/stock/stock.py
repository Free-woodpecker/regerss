# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:38:28 2020
获取指定股票历史信息
@author: 爬上阁楼的鱼
"""


import baostock as bs
import pandas as pd

def GetStockInfoDays(bs, code, end_date='2020-6-10', start_date='2000-01-01'):
    # exit
    rs = bs.query_history_k_data_plus(code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
        start_date, end_date,
        frequency="d", adjustflag="3")

    if rs.error_code == 0:
        a = '\nGetStockInfoDays 查询错误  错误代码: ' + rs.error_code
        a += '错误信息 : ' + rs.error_msg;
        raise Exception(a)
    else:
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        return pd.DataFrame(data_list, columns=rs.fields)





# lg = bs.login()

# result = GetStockInfoDays(bs, 'sh.600184')

# #### 结果集输出到csv文件 ####   
# #result.to_csv("history_A_stock_k_data.csv", index=False)

# #### 登出系统 ####
# bs.logout()


