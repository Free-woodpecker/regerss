# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:47:07 2020

@author: 爬上阁楼的鱼
"""
import sys
import tushare as ts 
import pandas as pd


df = ts.get_hist_data('600184',start='2013-06-04',end='2020-06-5')

#print(df)

df.to_excel(r'C:\123.xlsx')

df.to_csv(r'C:\123.csv')

#fh = open('666.txt', 'w', encoding='utf-8')
#fh.write(json.dumps(df))
#fh.close()


print('over')