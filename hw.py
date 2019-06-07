#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:14:34 2019

@author: sbaek
"""
import numpy as np
from scipy.optimize import minimize
import gd

# Least Squares function
def LeastSquares(x,A,b):
    return np.linalg.norm(A@x-b)**2

# gradient  
def grad_LeastSquares(x,A,b):
    return 2*((A.T@A)@x-A.T@b)

# size of matrix A (m by n) and b (m by 1)    
m=100
n=10

# for random number generation
mu=0
sig=1
    
# create matrices for least squares function
A=np.random.normal(mu,sig,(m,n))
b=np.random.normal(mu,sig,(m))

# initial point
x0=np.random.normal(mu,sig,(n))

# built-in minimize function
res=minimize(fun=LeastSquares,x0=x0,args=(A,b))
print('solution from minimize: message: ', res.message)
print('solution from minimize: success:', res.success)
print('solution from minimize: solution x:', res.x)


# your implementation of gradient descent
x=gd.min_gd(fun=LeastSquares,x0=x0,grad=grad_LeastSquares,args=(A,b))
print('solution from min_gd:', x)

# show error between built-in and your solutions
print('error :',np.linalg.norm(x-res.x))

 
