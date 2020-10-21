# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:47:35 2020

@author: xing
"""

import numpy as np
import matplotlib.pyplot as mp


x = np.arange(-0.005, 0.025, 0.001)

y = (np.tanh(120 * (x - 0.015)) + 1) * 0.45 + 0.05

mp.plot(x, y, lw = 1.0)




mp.show()


