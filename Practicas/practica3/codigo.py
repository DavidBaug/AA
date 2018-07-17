# -*- coding: utf-8 -*-
"""
Created on Thu May 17 22:44:53 2018

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.preprocessing import Normalizer, StandardScaler, QuantileTransformer, PolynomialFeatures

from sklearn.pipeline import Pipeline

from time import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


# LOAD DATA

def load_airfoil():
    X = np.load("./datos/airfoil_self_noise_X.npy")
    y = np.load("./datos/airfoil_self_noise_y.npy")
    
    if(X.shape[0]>10000):
        perc = 0.10
    else:
        perc = 0.20
        
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=perc, random_state=22, shuffle=True)
            
    return X_train, X_test, y_train, y_test
    
    
    
def load_digits():
    X_train = np.load("./datos/optdigits_tra_X.npy")
    X_test = np.load("./datos/optdigits_tes_X.npy")
    
    y_train = np.load("./datos/optdigits_tra_y.npy")
    y_test = np.load("./datos/optdigits_tes_y.npy")
    
    return X_train, y_train, X_test, y_test

###############################################################################
    


# Preprocesado de los datos de regresion
# Normalizar ,Estandarizar o Transformar características a una distribución
# Actualización de características al orden polinómico n, (a, b, a^2, ab, b^2....a^n)

def preprocessing_r(X_train, X_test, n):
       
    pipe = Pipeline([ ('Norm',Normalizer()),('QT', QuantileTransformer()),
                     ('Scale',StandardScaler()),
                     ('Pol',PolynomialFeatures(degree=n, include_bias=True))])
    
    pipe.fit(X_train)
    
    X_train = pipe.transform(X_train)
    X_test = pipe.transform(X_test)
    
    return X_train, X_test


# Preprocesado de los datos de clasificación
# Normalizar ,Estandarizar o Transformar características a una distribución

def preprocessing_c(X_train, X_test):
       
    pipe = Pipeline([ ('Norm',Normalizer()),('QT', QuantileTransformer()),
                     ('Scale',StandardScaler()),])
    
    pipe.fit(X_train)
    
    X_train = pipe.transform(X_train)
    X_test = pipe.transform(X_test)
    
    return X_train, X_test

###############################################################################
###############################################################################
###############################################################################


    
# Ajuste de modelo usando regresión logística
# A partir de unos parámetros prefijados usamos GridSearchCV para escoger el modelo
# que mejor puntuación obtenga
# Se devuelve el modelo estimado
def logistic_regression(X, y, folds):
    
    #Logistic Regression
    
    hyper= [{'penalty': ['l1','l2'], 'C': [1, 10, 100, 1000, 10000, 100000], 'max_iter':[15]}]
    
    print("Estimando parámetros para regresión logística...\n")
    
    clf = GridSearchCV(linear_model.LogisticRegression(), hyper, cv=folds, scoring='accuracy')
    
    inicial = time()
    clf.fit(X, y)
    
    final = time()
    
    print("Execution time : ", final-inicial,'s')
    
    print("Best parameters: ", clf.best_params_)
    
    print("Ein: ", clf.score(X,y))
    
    return clf, clf.score(X,y)



# Ajuste de modelo usando gradiente descendente estocástico
# A partir de unos parámetros prefijados usamos GridSearchCV para escoger el modelo
# que mejor puntuación obtenga
# Se devuelve el modelo estimado
def sgd_classifier(X, y, folds):
    
    # SGD Classifier
    
    hyper= [{'penalty': ['l1','l2','elasticnet'], 'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 'max_iter':[15]}]
    
    print("Estimando parámetros para clasificador sgd...\n")
    
    clf = GridSearchCV(linear_model.SGDClassifier(), hyper, cv=folds, scoring='accuracy')
    
    inicial = time()
    clf.fit(X, y)
    final = time()
    
    print("Execution time : ", final-inicial,'s')

    
    print("Best parameters: ", clf.best_params_)
    
    print("Ein: ", clf.score(X,y))
    
    return clf, clf.score(X,y)



    
def classification():
    
    X_train, y_train, X_test, y_test = load_digits()
    
    if(X_train.shape[0] > 10000):
        folds = 8
    else:
        folds = 5
    
    
#    X_train, X_test = preprocessing_c(X_train, X_test)

    ##############################
    print()
    print("LOGISTIC REGRESSION\n")
    
    lr_clf, lr_score = logistic_regression(X_train, y_train, folds)
    
   
    
    ##############################
    print()
    print("SGD CLASSIFIER\n")
    
    sgd_clf, sgd_score = sgd_classifier(X_train, y_train, folds)

 
    print('---------------------------------')

    print()
    print('sgd',sgd_score)
    print('lr',lr_score)
    if( sgd_score > lr_score):
        print('Best classifier: sgd')
        print('Params: ',sgd_clf.best_params_)
        y_pred = sgd_clf.predict(X_test)
    
        print('Eout: ',sgd_clf.score(X_test, y_test))
        print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))
    else:
        print('Best classifier: logistic regression')
        print('Params: ',lr_clf.best_params_)
        y_pred = lr_clf.predict(X_test)
        
        print('Eout: ',lr_clf.score(X_test, y_test))
        print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))
    
    
    
###############################################################################
###############################################################################
###############################################################################
    


# Ajuste de modelo usando ridge
# A partir de unos parámetros prefijados usamos GridSearchCV para escoger el modelo
# que mejor puntuación obtenga
# Se devuelve el modelo estimado
def ridge(X, y, folds):
    
    # Ridge
    
    hyper= [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}]
    
    print("Estimando parámetros para regresor ridge...\n")
    
    clf = GridSearchCV(linear_model.Ridge(), hyper, cv=folds, scoring='r2')
    
    inicial = time()
    clf.fit(X, y)
    final = time()
    
    print("Execution time : ", final-inicial,'s')

    
    print("Best parameters: ", clf.best_params_)
    
    print("Ein: ", clf.score(X,y))
    
    return clf, clf.score(X,y)

    
def regression():

    X_train, X_test, y_train, y_test = load_airfoil()
        
    if(X_train.shape[0] > 10000):
        folds = 20
    else:
        folds = 8

    models = []
    scores = []
    polynomials = []

    degrees = [1,2,3,4,5,6,7,8,9,10]

    for i in degrees:
        X_train, X_test, y_train, y_test = load_airfoil()
        X_train, X_test = preprocessing_r(X_train, X_test, i)
    
        print()
        print("POLYNOMIAL DEGREE: ",i)
    
        
        print()
        print("RIDGE\n")
        
        clf, score = ridge(X_train, y_train, folds)
        
        
        models.append(clf)
        scores.append(score)
        polynomials.append(i)
    
    
    
    best_score = np.max(scores)
    
    pos = np.where(best_score==scores)
    
    X_train, X_test, y_train, y_test = load_airfoil()
    X_train, X_test = preprocessing_r(X_train, X_test, polynomials[pos[0][0]])
    
    print('---------------------------------')
    print()
    print('Best polynomial degree: ', polynomials[pos[0][0]])
    print('Best classifier:',models[pos[0][0]])

    print("Eout: \n",models[pos[0][0]].score(X_test, y_test))
  
    
    
    
    
classification()
input("Press Enter to continue...\n\n") 
regression()   
    
    