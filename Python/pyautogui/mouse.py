# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:17:46 2020

@author: 爬上阁楼的鱼
"""

import pyautogui as auto


import time


while 1:
    time.sleep(5)
    
    auto.moveTo(x = 700, y = 700, tween=auto.linear )
    
    time.sleep(5)

    auto.moveTo(x = 700, y = 700, tween=auto.linear )
