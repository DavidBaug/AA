# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:01:19 2018

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from time import time

from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn import neural_network

from sklearn.preprocessing import label_binarize

from sklearn import ensemble

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# LOAD DATA

def load_data():
    X_train = np.loadtxt("datos/X_train.txt")
    y_train = np.loadtxt("datos/y_train.txt")
    
    X_test = np.loadtxt("datos/X_test.txt")
    y_test = np.loadtxt("datos/y_test.txt")
    
    
    return X_train, X_test, y_train, y_test



def grid_search(X,y,folds,alg):

    if (alg == 'sgd'):
        hyper= [{'penalty': ['l1','l2','elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'max_iter':[10], 'loss':['log','perceptron']}]
        clf = GridSearchCV(linear_model.SGDClassifier(), hyper, cv=folds, scoring='accuracy')

    elif (alg == 'svm'):
        hyper= [{'penalty':['l1','l2'], 'dual':[False],'C': [0.0001, 0.001, 0.01, 0.1,  1], 'max_iter':[10], 'loss':['squared_hinge']},
                 {'penalty':['l2'], 'C': [0.0001, 0.001, 0.01, 0.1,  1], 'max_iter':[10], 'loss':['hinge']}]
        clf = GridSearchCV(svm.LinearSVC(), hyper, cv=folds, scoring='accuracy')
        
    elif (alg == 'rf'):
        hyper= [{'n_estimators':[250], 'max_depth':[25], 'max_leaf_nodes':[20], 'n_jobs':[-1]}]
        clf = GridSearchCV(ensemble.RandomForestClassifier(), hyper, cv=folds, scoring='accuracy')
                
    elif (alg == 'boosting'):
        hyper= [{'n_estimators':[10], 'learning_rate':[0.001,0.01,0.1,1]}]
        clf = GridSearchCV(ensemble.AdaBoostClassifier(), hyper, cv=folds, scoring='accuracy')
        
    elif (alg == 'nn'):
        hyper= [{'hidden_layer_sizes':[9], 'alpha':[0.0001, 0.001, 0.01, 0.1], 'learning_rate_init':[0.001, 0.01, 0.1], 'max_iter':[200]}]
        clf = GridSearchCV(neural_network.MLPClassifier(), hyper, cv=folds, scoring='accuracy')        
        
    inicial = time()
    clf.fit(X, y)
    final = time()
    
    exec_time = final-inicial
    
    print("Execution time : ", exec_time,'s')

    
    print("Best parameters: ", clf.best_params_)
    
    print("Ein: ", 1-clf.score(X,y))
    
    print()
    print('--------------------------------------------------------')
    print()
    
    return clf, clf.score(X,y)
    

def classification():

    
    print()
    print("Proyecto Final\n")
    
    # Load Data
    
    X_train, X_test, y_train, y_test = load_data()

    algs = ['sgd','svm','rf','boosting','nn']   

#    algs=['rf']

    classifiers = []
    scores = []

    for alg in algs:
        print("Ajustando {}\n".format(alg))
        clf, score = grid_search(X_train, y_train, 5, alg)

        classifiers.append(clf)
        scores.append(score)
                
    
    # Binarizamos nuestro vector de etiquetas para poder calcular la curva roc
    labels = np.unique(y_train)
    
    y_train_b = label_binarize(y_train.copy(), classes=labels)
    y_test_b = label_binarize(y_test.copy(), classes=labels)
       
    n_classes= y_train_b.shape[-1]
    n_samples, n_features = X_train.shape
   
    
    best_clf = classifiers[scores.index(np.max(scores))]

    print("Best classifier: ",best_clf.best_estimator_)


    print("Eout: ", 1-best_clf.score(X_test,y_test))

    print()

#   Confusion matrix
    print("Confusion matrix:\n")
    y_predict_b = best_clf.predict(X_test)
    print(confusion_matrix(y_test, y_predict_b))


    # ROC Curve for best classifier
    y_predict_b = label_binarize(y_predict_b, classes=labels)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_predict_b[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic best classifier')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    
classification()