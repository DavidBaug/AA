# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:56:56 2018

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def simula_unif(N=2, dims=2, size=(0, 1)):
    m = np.random.uniform(low=size[0], 
                          high=size[1], 
                          size=(N, dims))
    
    return m


def label_data(x1, x2):
    y = np.sign((x1-0.2)**2 + x2**2 - 0.6)
    idx = np.random.choice(range(y.shape[0]), size=(int(y.shape[0]*0.1)), replace=True)
    y[idx] *= -1
    
    return y


#Generate data
def generateData():
    X=[]
    y=[]
    for i in range(1000):
        Xx = simula_unif(N=1000, dims=2, size=(-1, 1))
        yy = label_data(X[:, 0], X[:, 1])
    
        X.append(Xx)
        y.append(yy)
    
    return X,y

X,y = generateData()


#Averiguamos que clases hay en el dataset.
clases = np.unique(y)

print("Clases : {}".format(clases))

#Separamos los datos según su clase
#Estoy usando una forma de crear una lista con un bucle que permite hacer 
#python, puede sustituirse por un bucle for con un append a la lista.
X_class = [X[y==c_i] for c_i in clases]

#Separamos en train y test.
trainX_class = [Xi[:int(Xi.shape[0]*0.8)] for Xi in X_class]
testX_class = [Xi[int(Xi.shape[0]*0.8):] for Xi in X_class]


#Calculamos el nuevo tamaño.
sizes_train = [tX.shape[0] for tX in trainX_class]
sizes_test = [tX.shape[0] for tX in testX_class]

#Concatenamos
trainX = np.concatenate(trainX_class, axis=0)
testX = np.concatenate(testX_class, axis=0)

#Creamos trainY y testY
trainY = np.zeros(trainX.shape[0], y.dtype)
testY = np.zeros(testX.shape[0], y.dtype)
pos_train = pos_test = 0

#El comando zip permite empaquetar listas de la misma longitud para recorrerlas
#a la vez.
for c_i, size_train, size_test in zip(clases, sizes_train, sizes_test):
    end_train = pos_train+size_train
    end_test = pos_test+size_test
    
    trainY[pos_train:end_train] = c_i
    testY[pos_test:end_test] = c_i

    pos_train = end_train
    pos_test = end_test

#Eliminamos lo que sobra (no es necesario).
del X
del y
del sizes_train
del sizes_test
del pos_train
del pos_test

def function(data):
    return    np.sign((data[-2]-0.2)**2 + data[-1]**2 - 0.6)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Stochastic Gradient Descent
def sgd(model, X_train, y_train, iters, minibatch_size, n):
    
    tam = len(X_train)
    Error = []
    err = []
    
    for i in range(iters):
        print('Iteration {}'.format(i))

        aux = list(zip(X_train,y_train))
        
        random.shuffle(aux)

        X_train, y_train = zip(*aux) #Random datos
        
        for j in range(0, tam, minibatch_size): #For minibatch
            X = X_train[j:j+minibatch_size]
            y = y_train[j:j+minibatch_size]
            
            model, err = minibatch_step(model, X, y, n)
            Error += err.copy()
                    
    return model, Error

# w = w - E(h(x)-y)
def minibatch_step(w, X_train, y_train, n):
    
    E = []
    
    for X,y in zip(X_train, y_train):
        w -= n*gradient(w,X,y)
        err = error(w,X,y)

        E.append(err)
        
        """
        if(err < 1e-3):
            print("Error = {}".format(err))
            break
        """
        
    return w, E


def gradient(w, X, y):
    #print (np.float64(2/n)*X*((w.dot(X))-y))
    return np.float64(2/len(X))*X*((w.dot(X))-y)
    

def classificationError(w, X, y):
    h = np.sign(np.dot(X,w))
    
    fails = np.int(0)
    
    for h, y in zip(h,y):
        if (h != y):
            fails += 1
    
    print ("De un conjunto de datos de {} elementos ha fallado en predecir {}".format(len(X),fails))
    print ("ErrorClasificacion = {}".format((fails/len(X))))


#Error w*x = y
def error(w, X, y):
#    res = w.dot(X)
    res = np.dot(X,w)

    error = (res-y)**2
    
    #print("Error = {}".format(error))
    
    return error


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

minibatch_size = np.int(np.ceil(np.log2(len(trainX))*5))
max_iters = 3000

trainX = [[np.float64(1)]+list(tup) for tup in trainX]
testX = [[np.float64(1)]+list(tup) for tup in testX]

trainX = np.array(trainX)
trainY = np.array(trainY,np.float64)
testX = np.array(testX,np.float64)
testY = np.array(testY,np.float64)

n = np.float64(1e-1)

E = []
values = [(np.array([1,1,1],np.float64)) for x in trainX]

values, E = sgd(values, trainX, trainY, max_iters, minibatch_size, n)

print ("stochastic gradient descent -> Minibatch size : {}".format(minibatch_size))

print("w = {}".format(values))

print ("Learning rate : {}".format(n))

ErrorTrain = np.float64(np.mean(E))
print("Error train = {}\n".format(ErrorTrain))

print ("Error de clasificación en el train")
classificationError(values, trainX, trainY)


print ("")

ErrorTest = np.float64(np.mean(error (values, testX, testY)))
print("Error test = {}\n".format(ErrorTest))

print ("Error de clasificación en el test")
classificationError(values, testX, testY)