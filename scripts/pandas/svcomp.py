#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 12:03:44 2018

@author: ankit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset2 = pd.read_csv('trainXY.csv', delimiter = ' ')

#Add code for X_test after running Yulia's tool
#feature matrix
X_train = dataset2.iloc[:,1:-1].values

#label vector
y_train = dataset2.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from ml import Algorithms
algo = Algorithms()

#Random Forest Algorithm
y_pred = algo.rand_forest(X_train, y_train, X_test)
print(y_pred)