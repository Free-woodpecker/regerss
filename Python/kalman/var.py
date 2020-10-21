# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:14:51 2020

@author: 爬上阁楼的鱼
"""

import csv
import numpy as np
import matplotlib.pyplot as mp


M = 2
biasDiff = []
num = 0

a = []
i = 0

with open('var.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))
    
    for row in reader:
        a.append(row[0])

data = np.asarray(a, dtype =  float)

print(np.var(data))
















