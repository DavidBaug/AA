#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Ejercicio de clase. En este ejercicio se nos pide que:
    Leamos la base de datos de iris que hay en scikit-learn.
    Obtengamos de ella las características (datos de entrada X) y la clase (y).
    Nos quedemos con las dos primeras características (2 primeras columnas 
    de X).
    Separar en train 80% y test 20% aleatoriamente conservando la proporción de 
    elementos en cada clase tanto en train como en test.
"""

#Importamos paquetes necesarios.
import numpy as np
from sklearn import datasets

#Leemos el dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

#Nos quedamos con las 2 primeras características.
X = X[:, :2]

#Aleatorizamos los datos
#Voy a crear un vector de índices, aleatorizarlo y usarlo para indexar X e y.
idx = np.arange(0, X.shape[0], dtype=np.int32)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

#Averiguamos que clases hay en el dataset.
clases = np.unique(y)

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

print('Done!')
 
