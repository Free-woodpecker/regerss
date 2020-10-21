# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:30:18 2020

@author: 爬上阁楼的鱼
"""
import datetime
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
from sqlalchemy import Column, Integer, String, Float, DateTime
class Number(Base):
  # 表名称
  __tablename__ = 'Number'
  # news表里id字段
  Id = Column(Integer, primary_key=True, autoincrement=True)
  Num = Column(Float, nullable=False)




class sh000001(Base):
  # 表名称
  __tablename__ = 'day.sh.000001'
  # news表里id字段
  id = Column(Integer, primary_key=True, autoincrement=True)
  date = Column(DateTime, nullable=False)
  open = Column(Float, nullable=False)
  close = Column(Float, nullable=False)
  high = Column(Float, nullable=False)
  low = Column(Float, nullable=False)
  pctChg = Column(Float, nullable=False)
  

