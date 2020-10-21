# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:34:58 2020

@author: 爬上阁楼的鱼
"""

import numpy as np
class moments:
    
    area = 0
    
    m00 = 0
    m10 = 0
    m20 = 0
    m30 = 0
    m01 = 0
    m02 = 0
    m03 = 0
    m11 = 0
    m12 = 0
    m21 = 0
    
    mu02 = 0
    mu03 = 0
    mu11 = 0
    mu12 = 0
    mu20 = 0
    mu21 = 0
    mu30 = 0
    
    nu02 = 0
    nu03 = 0
    nu11 = 0
    nu12 = 0
    nu20 = 0
    nu21 = 0
    nu30 = 0
    
    hu1 = 0
    hu2 = 0
    hu3 = 0
    hu4 = 0
    hu5 = 0
    hu6 = 0
    hu7 = 0
    
    
    def __init__(self, array = np.random.rand(4,2), data = 1):
        
        # m10 = self.calculateMoment(array = array, data = data, p = 1, q = 0)
        
        if data == 1:
            data = np.ones(len(array))
            
        self.m00 = np.sum(data)
        self.area = len(array)
        
        # 区域矩
        self.m10 = np.sum(array[...,0]**1 * data)
        self.m20 = np.sum(array[...,0]**2 * data)
        self.m30 = np.sum(array[...,0]**3 * data)
        self.m01 = np.sum(array[...,1]**1 * data)
        self.m02 = np.sum(array[...,1]**2 * data)
        self.m03 = np.sum(array[...,1]**3 * data)
        self.m11 = np.sum(array[...,0]**1 * array[...,1]**1 * data)
        self.m12 = np.sum(array[...,0]**1 * array[...,1]**2 * data)
        self.m21 = np.sum(array[...,0]**2 * array[...,1]**1 * data)
        
        # 中心矩
        self.mu02 = np.sum((self.m01 - array[...,1])**2 * data)
        self.mu03 = np.sum((self.m01 - array[...,1])**3 * data)
        self.mu11 = np.sum((self.m10 - array[...,0])**1 * (self.m01 - array[...,1])**1 * data)
        self.mu12 = np.sum((self.m10 - array[...,0])**1 * (self.m01 - array[...,1])**2 * data)
        self.mu21 = np.sum((self.m10 - array[...,0])**2 * (self.m01 - array[...,1])**1 * data)
        self.mu20 = np.sum((self.m10 - array[...,0])**2 * data)
        self.mu30 = np.sum((self.m10 - array[...,0])**3 * data)
        
        # Hu矩 归一化中心矩
        self.nu02 = self.mu02/(self.m00**2)
        self.nu03 = self.mu03/(self.m00**2.5)
        self.nu11 = self.mu11/(self.m00**2)
        self.nu12 = self.mu12/(self.m00**2.5)
        self.nu20 = self.mu20/(self.m00**2)
        self.nu21 = self.mu21/(self.m00**2.5)
        self.nu30 = self.mu30/(self.m00**2.5)
        
        self.hu1 = self.nu20 + self.nu02
        self.hu2 = (self.nu20 - self.nu02)**2 + 4*self.nu11**2
        self.hu3 = (self.nu20 - 3*self.nu12)**2 + 3*(self.nu12 - self.nu03)**2
        self.hu4 = (self.nu30 + self.nu12)**2 + (self.nu21 + self.nu03)**2
        self.hu5 = (self.nu30 + 3*self.nu12)*(self.nu30 + self.nu12)*((self.nu30 + self.nu12)**2 - 3*(self.nu12 + self.nu03)**2) - (3*self.nu21 - self.nu03)*(self.nu21 + self.nu03)*(3*(self.nu30 + self.nu12)**2 - (self.nu21 + self.nu03)**2)
        self.hu6 = 0
        self.hu7 = 0
    
    
    def calculateMoment(array, data = 1, p=0, q=0):
        if data == 1:
            data = np.ones(len(array))
        # x, y = array.shape
        return np.sum(array[...,0]**p * array[...,1]**q * data)


a = np.array([[1,1],[1,2],[2,1],[2,2]])
b = np.random.rand(4,2)
M = moments(a)
    
    