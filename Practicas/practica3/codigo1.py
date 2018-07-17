# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:56:17 2018

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.preprocessing import Normalizer, StandardScaler, QuantileTransformer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


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

def preprocessing(X_train, X_test, n):
    
    if (n == 1):
        nrm = Normalizer(norm='l2')

        X_train = nrm.fit_transform(X_train)
        X_test = nrm.fit_transform(X_test)
    elif(n == 2):
        scl = StandardScaler()
        
        X_train = scl.fit_transform(X_train)
        X_test = scl.fit_transform(X_test)
    elif(n == 3):
        qt = QuantileTransformer()
        
        X_train = qt.fit_transform(X_train)
        X_test = qt.fit_transform(X_test)

    
    
    return X_train, X_test    


###############################################################################
# Split X in k folds and train - validate
# Using lasso and ridge, fit a model
def cross_validation(X,y,funcion, penal=None):

    #print(X.shape[0])
    
    if(X.shape[0] > 10000):
        folds = 7
    else:
        folds = 5
    
    kf = KFold(n=X.shape[0], n_folds=folds, random_state=1)
    
    models = []
    k_scores = []
    
    if(funcion == linear_regression):
        for train, val in kf:
            X_train, y_train = X[train], y[train]
            X_val, y_val = X[val], y[val]
               
            reg = funcion(X_train, y_train)
    
            sc = reg.score(X_val, y_val)
                
            models.append(reg)
            k_scores.append(sc)
    else:
        # Hyperparameters
        for j in range(6):
            if(funcion == sgd):        
                m=1/(10**j)            
            else:
                m = 10**j
            
            for train, val in kf:
                X_train, y_train = X[train], y[train]
                X_val, y_val = X[val], y[val]
                   
                reg = funcion(X_train, y_train, m, penal)
        
                sc = reg.score(X_val, y_val)
                    
                models.append(reg)
                k_scores.append(sc)
                    
    # Escogemos mejor score de los k-folds
    k_scores = np.asarray(k_scores)

    best_score = np.max(k_scores)
    pos = np.where(best_score==k_scores)
    
    print("Best score validation:",k_scores[pos[0][0]])

    return models[pos[0][0]]

    
###############################################################################

# Función que nos ajusta usando regresión logística y que nos devuelve modelo
def logistic_regression(X_train, y_train, a, penal):
    clf = OneVsRestClassifier(linear_model.LogisticRegression(penalty=penal,fit_intercept=True, C=a, max_iter=10))
        
    clf.fit(X_train, y_train)
    
    return clf

# Función que nos ajusta usando gradiente descendente estocástico y que nos devuelve modelo
def sgd(X_train, y_train , a, penal):
    clf = OneVsRestClassifier(linear_model.SGDClassifier(penalty=penal, alpha=a, fit_intercept=True, max_iter=10))
        
    clf.fit(X_train, y_train)
    
    return clf

# Función que nos ajusta usando gradiente descendente estocástico y que nos devuelve modelo
def sgd_r(X_train, y_train , a, penal):
    clf = linear_model.SGDRegressor(penalty=penal, alpha=a, fit_intercept=True, max_iter=10)
        
    clf.fit(X_train, y_train)
    
    return clf

# Función que nos ajusta usando gradiente descendente estocástico y que nos devuelve modelo
def ridge(X_train, y_train, a, penal=None):
    clf = linear_model.Ridge(alpha=a, fit_intercept=True, max_iter=10)
        
    clf.fit(X_train, y_train)
    
    return clf

# Función para ajustar un modelo usando regresión lineal
def linear_regression(X_train, y_train):
    clf = linear_model.LinearRegression(fit_intercept=True)
        
    clf.fit(X_train, y_train)
    
    return clf



###############################################################################

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
   

# Función que nos calcula el mejor modelo ajustado con distintas penalizaciones    
def digits(X_train, y_train, X_test, y_test, classifier, preproc):
    
    np.random.seed(0)
    
    # CLASSIFIER 
    
    print()
    print("{} Classifier ".format(classifier))    
    print()
    
    
    
    if(classifier != sgd):
        penalties = ['l1','l2']
    else:
        penalties = ['l1','l2','elasticnet']
    
    models = []
    scores = []
    y_s = []
    
    for p in penalties:    
        print("Penalty: ", p)
        model = cross_validation(X_train,y_train,classifier, p)
    
        y_pred = model.predict(X_test)
        models.append(model)
   
        y_s.append(y_pred)
    
        scores.append(accuracy_score(y_test,y_pred))
    
        print("Accuracy score using {}: {}".format(classifier, scores[-1]))
        print()
    
        
    best_score = np.max(scores)
    pos = np.where(best_score==scores)
    
    
    print("Chosen model: ",models[pos[0][0]])
    #print("Classification score using {}: \n{}".format(classifier,classification_report(y_test,y_s[pos[0][0]])))
    #print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))
    
    print()
    
    return models[pos[0][0]], best_score
        
    
def compare_classification():
    
    
    X_train, y_train, X_test, y_test = load_digits()
   
    classifiers = [sgd, logistic_regression]
    
    models = []
    scores = []
    prep = []
    
    pre = [1,2,3]
    
    for p in pre:
        X_train, X_test = preprocessing(X_train, X_test, p)
        
        for clf in classifiers:
            model, score = digits(X_train, y_train, X_test, y_test, clf, p)
        
            models.append(model)
            scores.append(score)
            prep.append(p)

    best_score = np.max(scores)
    pos = np.where(best_score==scores)
    
    print("Used {} preprocessing.".format(prep[pos[0][0]]))
    print("Best Classifier: ", models[pos[0][0]])
    print("Accuracy score: ", scores[pos[0][0]])


###############################################################################

def airfoil(X_train, y_train, X_test, y_test, regression, preproc):
    
    np.random.seed(0)
    
    # CLASSIFIER 
    
    print()
    print('--------------------------------------------------------------')
    print("{} Regression ".format(regression))    
    print()

    
    if(regression == ridge):
        penalties = ['l1','l2']
        
        models = []
        y_s = []
        scores = []
        
        for p in penalties:    
            print("Penalty: ", p)
            model = cross_validation(X_train,y_train, ridge, p)
        
            y_pred = model.predict(X_test)
            models.append(model)
       
            y_s.append(y_pred)
        
            scores.append(r2_score(y_test,y_pred))
        
            print("r2_score using {}: {}".format(regression, scores[-1]))
            print()
            
        best_score = np.max(scores)
        pos = np.where(best_score==scores)
        
        
        print("Chosen model: ",models[pos[0][0]])
        print("r2_score using {}: \n{}".format(regression,best_score))
        #print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))
        
        print()
    
        return models[pos[0][0]], best_score
    
    else:
        model = cross_validation(X_train, y_train, linear_regression)
        y_pred = model.predict(X_test)
        
        score = r2_score(y_test,y_pred)
        
        print("Chosen model: ",model)
        print("r2_score using {}: \n{}".format(regression,score))
        #print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))
        
        print()
        
        return model, score

def compare_regression():
    
    X_train, y_train, X_test, y_test = load_digits()
       
    classifiers = [ridge, linear_regression, sgd_r]
    
    models = []
    scores = []
    prep = []
    
    pre = [1,2,3]
    
    for p in pre:
        X_train, X_test = preprocessing(X_train, X_test, p)
            
        for clf in classifiers:
            model, score = airfoil(X_train, y_train, X_test, y_test, clf, p)
        
            models.append(model)
            scores.append(score)
            prep.append(p)

    best_score = np.max(scores)
    pos = np.where(best_score==scores)
    
    print("Used {} preprocess.".format(prep[pos[0][0]]))
    print("Best regressor: ", models[pos[0][0]])
    print("r2_score: ", scores[pos[0][0]])

###############################################################################

#compare_classification()
    
compare_regression()