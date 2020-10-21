# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:36:08 2020

@author: xing
"""


#   python实现LU分解
 
import numpy as np
 
def my_LU(B):
    A = np.array(B)
    n = len(A)
    #print(A)
    
    L = np.zeros(shape=(n,n))
    U = np.zeros(shape=(n,n))
    
 
    for k in range(n-1):
        gauss_vector = A[:,k]
        gauss_vector[k+1:] = gauss_vector[k+1:] / gauss_vector[k]
        gauss_vector[0:k+1] = np.zeros(k+1)
        #print(gauss_vector) 
        L[:,k] = gauss_vector
        L[k][k] = 1.0
        #print(L)
        #print(A)
        for l in range(k+1,n):
            B[l,:] = B[l,:] - gauss_vector[l] * B[k,:]
        
        A = np.array(B)
    L[k+1][k+1] = 1.0
    U = A
    print(U)
    print(L)
    
    A1 =  np.linalg.inv(U).dot(np.linalg.inv(L))
    print(A1)
    

def main():
    A = np.array([[1599.280,6688.040,29064.400],
                  [0.254,-96.571,-681.667],
                  [0.071,0.694,-21.141]])
    my_LU(A)
        
     
main()
