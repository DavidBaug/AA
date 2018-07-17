# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 16:21:08 2018

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt


X_train = np.load("./X_train.npy")
X_test = np.load("./X_test.npy")

y_train = np.load("./y_train.npy")
y_test = np.load("./y_test.npy")


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
def plot_data(X, y, w,tipo):
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
    plt.title('Pseudo Inversa'+tipo)
    plt.show()
 



#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


def setData(X_train, y_train, X_test, y_test):
    
    X_train.tolist()    
    y_train.tolist()
    X_test.tolist()
    y_test.tolist()

    X_train = X_train[(y_train==5) + (y_train==1)]
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
    
    print("Classes = ",np.unique(y_train))

    
    return X_train, y_train, X_test, y_test


def pseudoInverse(X_train, y_train):
    
    # w = (X_transpose * X)^-1   * X_transpose * y
    
    Xtranspose = np.transpose(X_train.copy())
    
    Xt = Xtranspose.copy()
    
    Xx = np.dot(Xt,X_train)
    
    Xx = np.linalg.inv(Xx)
    
    
    return np.dot(np.dot(Xx,Xt),y_train)


def error(w, X, y):
    res = np.dot(X,w)
    error = (res-y)**2
    
    return error


def classificationError(w, X, y):
    h = np.sign(np.dot(X,w))
    
    fails = np.int(0)
    
    for h, y in zip(h,y):
        if (h != y):
            fails += 1
    
    print ("De un conjunto de datos de {} elementos ha fallado en predecir {}".format(len(X),fails))

#calcular pseudoinversa
X_train, y_train, X_test, y_test = setData(X_train, y_train, X_test, y_test)

w = pseudoInverse(X_train, y_train)

print("w = {}".format(w))


#Calcular err train
Err = error(w, X_train,y_train)

ErrorTrain = np.float64(np.mean(Err))

print ("Error train = {}\n".format(ErrorTrain))

print ("Error de clasificación en el train")
classificationError(w, X_train, y_train)

#Quitar columna 1 - Cargamos la original
#X_train = np.load("./X_train.npy")
#X_test = np.load("./X_test.npy")

plot_data(X_train, y_train, w, ' - train')


print("")

#Calcular err test
Err = error(w, X_test,y_test)

ErrorTest = np.float64(np.mean(Err))

print ("Error test = {}\n".format(ErrorTest))


print ("Error de clasificación en el test")
classificationError(w, X_test, y_test)

plot_data(X_test, y_test, w, ' - test')



