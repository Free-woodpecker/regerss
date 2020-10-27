# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 08:33:37 2020

@author: 爬上阁楼的鱼
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt

sampling_rate = 1024#采样频率
t = np.arange(0,1.0,1.0/sampling_rate)
f1 = 100#频率
f2 = 200
f3 = 300
data = np.piecewise(t,[t<1,t<0.8,t<0.3],
                    [lambda t : np.sin(2*np.pi *f1*t),
                     lambda t : np.sin(2 * np.pi * f2 * t),
                     lambda t : np.sin(2 * np.pi * f3 * t)])
wavename = "cgau8"
totalscal = 10000
fc = pywt.central_frequency(wavename)#中心频率
cparam = 2 * fc * totalscal
scales = cparam/np.arange(totalscal,1,-1)
[cwtmatr, frequencies] = pywt.cwt(data,scales,wavename,1.0/sampling_rate)#连续小波变换
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(t, data)
plt.xlabel(u"time(s)")
plt.title(u"300Hz 200Hz 100Hz Time spectrum")
plt.subplot(212)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)
plt.show()