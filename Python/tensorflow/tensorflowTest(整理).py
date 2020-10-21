# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:24:57 2020

@author: 爬上阁楼的鱼
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd

# 下载数据集到本地
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
dataset_path = keras.utils.get_file("auto-mpg.data", url)


column_names = ['MPG','气缸','排量','马力','重量','加速度', '年份', '产地']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
# 数据 列名 ....


dataset = raw_dataset.copy()
# 查看前20条数据
print(dataset.head(20))

#dataset.to_csv(r'D:\321.csv')

