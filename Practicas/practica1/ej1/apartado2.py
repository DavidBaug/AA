# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:54:01 2018

@author: David
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

w= [np.float64(1),np.float64(1)]

E = np.float64(0)

n1 = np.float64(0.01)
n2 = np.float64(0.1)

n_iters = 2000000
stop = 50

#función de coste que admite cualquier f(x,y)
def cost(function,w):
    return (function(w[-2],w[-1]))

def function(x,y):
    return np.float64((x - 2)**2 + 2*(y + 2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))

#derivada parcial de x
def gradientU(x,y):
    return np.float64(4*np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*x) + 2*(x - 2))
    
#derivada parcial de y
def gradientV(x,y):
    return np.float64(4*np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + y + 2)
    

#Gradiente de la función
def gradient(data, w):
    grU = gradientU(w[-2],w[-1])
    grV = gradientV(w[-2],w[-1])
    
    return np.array([grU,grV],np.float64)

#Función iterativa para el cálculo de los valores mínimos/máximos de la función
def gradientDescent(iters, stop, n, w, gradient, function, graph):
   
    #Inicializar lista de Error y lista con iteraciones
    E = [cost(function,w)]

    #Vector donde guardamos todos los valores
    wValues = []
    
    wValues.append(w[-2])
    wValues.append(w[-1])
    
    for i in range (iters):
               
        w -= n*gradient(0,w)
        
        E.append(cost(function,w))
        
        #Aniadir valores de w
        wValues.append(w[-2])
        wValues.append(w[-1])

         
        if (E[len(E)-1] < 10**-50):
            print("Error: {} -> iter: {}".format(E[len(E)-1],i))
            print("Sol (u,v) = ({},{}) with {} iterations and n = {}.\n".format(w[0],w[1],i,n))
            break
        
        if (i == stop):
            print("Error: {} -> iter: {}".format(E[len(E)-1],i))
            print("Max iters({}) with n = {} -> Sol (u,v) = ({},{})\n".format(stop,n,w[0],w[1]))
            break
    
    # plot2d iters + E
    if(graph):    
        x = range(0,len(E))
        plt.plot(x, E, label='n = {}'.format(n))   
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.legend()
        plt.title('Gradient')
    
    return wValues



gradientDescent(n_iters, stop, n1, w, gradient, function, bool(1))
gradientDescent(n_iters, stop, n2, w, gradient, function, bool(1))

plt.show()


""" Segundo apartado """

print(" ------------------------------------------------- ")

stop = 200000

w1 = [np.float64(2.1),np.float64(-2.1)]
values1 = gradientDescent(n_iters, stop, n2, w1, gradient, function, bool(0))

w2 = [np.float64(3),np.float64(-3)]
values2 = gradientDescent(n_iters, stop, n2, w2, gradient, function, bool(0))

w3 = [np.float64(1.5),np.float64(1.5)]
values3 = gradientDescent(n_iters, stop, n2, w3, gradient, function, bool(0))

w4 = [np.float64(1),np.float64(-1)]
values4 = gradientDescent(n_iters, stop, n2, w4, gradient, function, bool(0))

tabla = [0,1,2,3]
tabla[0] = values1.copy()
tabla[1] = values2.copy()
tabla[2] = values3.copy()
tabla[3] = values4.copy()

#print(tabla[0][1])