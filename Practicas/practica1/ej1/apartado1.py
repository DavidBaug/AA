# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:54:01 2018

@author: David
"""
import numpy as np
import matplotlib.pyplot as plt


#   E(u, v) = (u^3 exp(v-2) - 4 v^3 exp -u)^2

#   2 (e^(-2 + v) u^3 - 4 e^(-u) v^3) (3 e^(-2 + v) u^2 + 4 e^(-u) v^3)
#   2 (e^(-2 + v) u^3 - 12 e^(-u) v^2) (e^(-2 + v) u^3 - 4 e^(-u) v^3)

np.random.seed(0)

w = np.array([1, 1], np.float64)
E = np.float64(0)

n = np.float64(0.04)


n_iters = 5000000
stop = 3000000

def cost(function,w):
    return (function(w[-2],w[-1]))
   
    
def function(x,y):
    return np.float64((x**3*np.exp(y-2) - 4*y**3*np.exp(-x))**2)

def gradientU(u,v):
    return np.float64(2*(np.exp(-2 + v)*u**3 - 4*np.exp(-u)*v**3)*(3*np.exp(-2 + v)*u**2 + 4*np.exp(-u)*v**3))
    
def gradientV(u,v):
    return np.float64(2*(np.exp(-2 + v)*u**3 - 12*np.exp(-u)*v**2)*(np.exp(-2 + v)*u**3 - 4*np.exp(-u)*v**3))

#Gradiente de la función
def gradient(data, w):
    grU = gradientU(w[len(w)-2],w[len(w)-1])
    grV = gradientV(w[len(w)-2],w[len(w)-1])
    
    return np.array([grU,grV],np.float64)


def gradientDescent(iters, stop, n, w, gradient, function):
   
    for i in range (iters):
        w -= n*gradient(0,w)
        E = cost(function,w)
        
        print("Error: {} -> iter: {}".format(E,i))
        
        if (E < (10**-14)):
            print("Sol (u,v) = ({},{}) with {} iterations.".format(w[0],w[1],i))
            break
        
        if (i == stop):
            print("Max iters({}) with n = {} -> Sol (u,v) = ({},{})".format(stop,n,w[0],w[1]))
            break


gradientDescent(n_iters, stop, n, w, gradient, function)