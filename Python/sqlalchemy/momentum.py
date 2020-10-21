# -*- coding: utf-8 -*-

"""
Created on Wed Jul  1 09:50:52 2020

@author: 爬上阁楼的鱼
"""

from sqlalchemy import create_engine
from Infos import sh000001
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as mp


# 登录数据库
engine = create_engine("mssql+pymssql://sa:521374335@localhost/Reason")
session = sessionmaker(engine)
mySession = session()

# 按 class 查询数据
result = mySession.query(sh000001).filter(sh000001.date > '2020-1-1').all()

i=0
last = 0
sums = 0
for su in result:
    sums += su.pctChg
    mp.plot(i, (su.close - 3000)*0.01, 'r.')
    mp.plot(i, su.pctChg, 'y.')         # 变化率
    # mp.plot(i, sums, 'b.')         # 变化率
    # mp.plot(i, su.pctChg - last, 'g.')  # 变化率的微分
    last = su.pctChg
    
    i+=1
    
    if i> 300:
        break

mp.show()



