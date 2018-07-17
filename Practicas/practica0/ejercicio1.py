#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:37:56 2018

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(0)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

x = X.copy()
y = Y.copy()

x = x[:,2:]

x_train, x_test, y_train, y_test = train_test_split(x,y)

lr = LinearRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test))