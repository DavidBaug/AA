# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 13:46:48 2018

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(0)

n = np.float64(0.1)

max_iters = np.int32(3000)

X_train = np.load("./X_train.npy")
X_test = np.load("./X_test.npy")

y_train = np.load("./y_train.npy")
y_test = np.load("./y_test.npy")

minibatch_size = np.int(np.ceil(np.log2(len(X_train))*5))

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


def coef2line(w):
    if(len(w)!= 3):
        raise ValueError('Solo se aceptan rectas para el plano 2d. Formato: [<a0>, <a1>, <b>].')
    
    a = -w[0]/w[1]
    b = -w[2]/w[1]
    
    return a, b


'''
    Pinta los datos con su etiqueta y la solución definida por w.
    X: Datos (Intensidad promedio, Simetría).
    y: Etiquetas (-1, 1).
    w: Lista o array de numpy (1d) con 3 valores. w[2] ha de ser el témino 
       independiente.
'''
def plot_data(X, y, w, tipo):
    #Preparar datos
    a, b = coef2line(w)
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = grid.dot(w)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$w^tx$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white', label='Datos')
    ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Solucion')
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel='Intensidad promedio', ylabel='Simetria')
    ax.legend()
    plt.title('Gradiente descendente estocástico ' + tipo)
    plt.show()
 



#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------



# Nums 1 y 5
def setData(X_train, y_train, X_test, y_test):
    
    X_train.tolist()    
    y_train.tolist()
    X_test.tolist()
    y_test.tolist()

    X_train = X_train[(y_train==5) + (y_train==1)] #XOR
    y_train = y_train[(y_train==5) + (y_train==1)]

    X_test = X_test[(y_test==5) + (y_test==1)]
    y_test = y_test[(y_test==5) + (y_test==1)]

    y_train[y_train==5] = -1
    y_test[y_test==5] = -1

    X_train = [list(tup)+[np.float64(1)] for tup in X_train]
    X_test= [list(tup)+[np.float64(1)] for tup in X_test]

    X_train = np.array(X_train,np.float64)    
    y_train = np.array(y_train,np.float64)
    X_test = np.array(X_test,np.float64)
    y_test = np.array(y_test,np.float64)
    
    
    return X_train, y_train, X_test, y_test


# Stochastic Gradient Descent
def sgd(model, X_train, y_train, iters, minibatch_size):
    
    tam = len(X_train)
    Error = []
    err = []
    
    print("Loading... \nPlease, do not turn off your computer")
    
    for i in range(iters):
        #print('Iteration {}'.format(i))

        aux = list(zip(X_train,y_train))
        
        random.shuffle(aux)

        X_train, y_train = zip(*aux) #Random datos
        
        for j in range(0, tam, minibatch_size): #For minibatch
            X = X_train[j:j+minibatch_size]
            y = y_train[j:j+minibatch_size]
            
            model, err = minibatch_step(model, X, y)
            Error += err.copy()
                    
    return model, Error

# w = w - E(h(x)-y)
def minibatch_step(w, X_train, y_train):
    
    E = []
    
    for X,y in zip(X_train, y_train):
        w -= 1e-4*gradient(w,X,y)
        err = error(w,X,y)

        E.append(err)
        
        """
        if(err < 1e-3):
            print("Error = {}".format(err))
            break
        """
        
    return w, E


def gradient(w, X, y):
    return np.float64(2/len(X))*X*((w.dot(X))-y)
    

def classificationError(w, X, y):
    h = np.sign(np.dot(X,w))
    
    fails = np.int(0)
    
    for h, y in zip(h,y):
        if (h != y):
            fails += 1
    
    print ("De un conjunto de datos de {} elementos ha fallado en predecir {}".format(len(X),fails))


#Error w*x = y
def error(w, X, y):
#    res = w.dot(X)
    res = np.dot(X,w)

    error = (res-y)**2
    
    #print("Error = {}".format(error))
    
    return error

Error = []
X_train, y_train, X_test, y_test = setData(X_train, y_train, X_test, y_test)
model = np.array([1, 1, 1], np.float64)
model, Error = sgd(model, X_train, y_train, max_iters, minibatch_size)


print ("stochastic gradient descent -> Minibatch size : {}".format(minibatch_size))

print("w = {}".format(model))

ErrorTrain = np.float64(np.mean(Error))
print("Error train = {}\n".format(ErrorTrain))

print ("Error de clasificación en el train")
classificationError(model, X_train, y_train)

plot_data(X_train, y_train, model, ' - Train')

print ("")

ErrorTest = np.float64(np.mean(error (model, X_test, y_test)))
print("Error test = {}\n".format(ErrorTest))

print ("Error de clasificación en el test")
classificationError(model, X_test, y_test)

plot_data(X_test, y_test, model, ' - Test')
























def pseudoInverse(X_train, y_train):
    
    # w = (X_transpose * X)^-1   * X_transpose * y
    
    Xtranspose = np.transpose(X_train.copy())
    
    Xt = Xtranspose.copy()
    
    Xx = np.dot(Xt,X_train)
    
    Xx = np.linalg.inv(Xx)
    
    
    return np.dot(np.dot(Xx,Xt),y_train)


