# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:56:17 2018

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.multiclass import OneVsRestClassifier

from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve


# Linear: Bayesian Ridge - Logistic Regression - Perceptron - SGD 


###############################################################################
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

def preprocessing(X_train, X_test):
    nrm = Normalizer(norm='l1')
    scl = StandardScaler()
    
    nrm.fit_transform(X_train)
    nrm.fit_transform(X_test)
    
    scl.fit_transform(X_train)
    scl.fit_transform(X_test)
    
    return X_train, X_test    


###############################################################################
# Split X in k folds and train - validate
# Using lasso and ridge, fit a model
def cross_validation(X,y,funcion, penal, folds=5):

    #print(X.shape[0])
    
    if(X.shape[0] > 10000):
        folds = 7
    
    kf = KFold(n=X.shape[0], n_folds=folds, random_state=1)
    
    models = []
    k_scores = []
    
    lasso_ridge = []
    
    best_index = []
    
    # Alpha parameters
    for j in range(6):
        if(funcion == sgd):        
            m=1/(10**j)            
        else:
            m = 10**j
        
        # Lasso parameters
        for i in range(7):
            n = 10**(-8+i)
            
            #print("Lasso alpha = ", n)
            
            for train, val in kf:
                X_train, y_train = X[train], y[train]
                X_val, y_val = X[val], y[val]
            
                #Lasso fit - L1
                index = lasso(X_train, y_train, n)
            
                # Remove w == 0
                X_train = np.delete(X_train, index, axis=1)
                X_val = np.delete(X_val, index, axis=1)
    
                # Model fit            
                reg = funcion(X_train, y_train, penal, m)
                #reg = logistic_regression(X_train, y_train, X_val, y_val)             
                
                sc = reg.score(X_val, y_val)
                
                #print("\tscore = ", sc)
                
                models.append(reg)
                k_scores.append(sc)
                lasso_ridge.append((n,m))
                best_index.append(index)
                
        
    k_scores = np.asarray(k_scores)

    best_score = np.max(k_scores)
    pos = np.where(best_score==k_scores)
    
    print("Best score validation:",k_scores[pos[0][0]])

    return models[pos[0][0]], lasso_ridge[pos[0][0]], best_index[pos[0][0]]
    


# Función que nos ajusta usando regresión logística y que nos devuelve modelo
def logistic_regression(X_train, y_train, penal, c):
    clf = OneVsRestClassifier(linear_model.LogisticRegression(penalty=penal,fit_intercept=True, C=c, max_iter=10))
        
    clf.fit(X_train, y_train)
    
    return clf

# Función que nos ajusta usando perceptron y que nos devuelve modelo
def sgd(X_train, y_train, penal, a):
    clf = OneVsRestClassifier(linear_model.SGDClassifier(penalty=penal, alpha=a, fit_intercept=True, max_iter=10))
        
    clf.fit(X_train, y_train)
    
    return clf


# Función para quitar variables inservubles
def lasso(Xt, yt, n):
    #Random seed
    np.random.seed(0)
        
    #print("X_train shape:", Xt.shape)
    
    # n = alpha lasso
    reg = linear_model.Lasso(alpha=n, fit_intercept=True)
    
    reg.fit(Xt, yt)
      
    w = reg.coef_
    
    #array con características irrelevantes 
    index = np.where(w <= 10**-5)

    return index




def digits_sgd():

    np.random.seed(0)
    
    # SGD
    
    print("SGD Classifier ")
    
    X_train, y_train, X_test, y_test = load_digits()
    
    
    model, param, index = cross_validation(X_train,y_train, sgd, 'l2')
              
#    X_test = np.delete(X_test, index, axis=1)
        
    y_pred = model.predict(X_test)
    
    print("alpha lasso: ", param[1])
    print("alpha ridge: ", param[0])
    print("Classification score using ridge: \n", classification_report(y_test,y_pred))
    print("Accuracy score using ridge: ", accuracy_score(y_test,y_pred))
    print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))
    
    print()
    
    
def digits_logistic_regression():
    
    np.random.seed(0)
    
    # LOGISTIC REGRESSION        
    
    print("Logistic Regression Classifier ")    
    
    X_train, y_train, X_test, y_test = load_digits()
    
    X_train, X_test = preprocessing(X_train, X_test)
    
    model, param, index = cross_validation(X_train,y_train,logistic_regression, 'l2')
              
#    X_test = np.delete(X_test, index, axis=1)
        
    y_pred = model.predict(X_test)
    
    print("alpha lasso: ", param)
    print("Classification score using logistic regression: \n", classification_report(y_test,y_pred))
    print("Accuracy score using logistic regression: ", accuracy_score(y_test,y_pred))
    print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))
    
    print()
        
    
def airfoil_():
    
    np.random.seed(0)
    
    print("assada")    
    
    X_train, y_train, X_test, y_test = load_airfoil()
    
    model, param, index = cross_validation(X_train,y_train)
              
    X_test = np.delete(X_test, index, axis=1)
        
    
    print("alpha lasso: ", param)
    print("Classification score using ridge: ", model.score(X_test, y_test))
    print()
    

    
digits_sgd()
digits_logistic_regression()

#load_airfoil()