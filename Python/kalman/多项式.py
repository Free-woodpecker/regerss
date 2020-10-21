# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:02:01 2020

@author: xing
"""

import numpy as np
import csv
# import math as mh
import matplotlib.pyplot as mp


biasDiff = []
num = 0

T = []
dT = []
bias = []
testx = []
y = []

i = 0
with open('20600.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))
    
    for row in reader:
        T.append(row[0])
        dT.append(row[1])
        bias.append(row[2])
        biasDiff.append(row[3])
        
        num += 1
        testx.append(num)

a = np.asarray(biasDiff, dtype =  float)
x = np.asarray(dT, dtype =  float)



y = x**2 * -1.45827 + x * 0.279347 + 0.00535631















