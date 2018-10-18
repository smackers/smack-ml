#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:53:45 2018

@author: ankit
"""

def accuracy(cm,m):    
    acc = sum(np.diag(cm))/m
    #print(acc)
    return acc*100

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/proj/SMACK/sv-benchmarks/c/'

#<tool 1> - features sent by Yulia, SVCOMP 2015
#<tool 2> - features generated using Yulia's tool, SVCOMP 2017
#loading data
df_2a = pd.read_csv('f_tool2a.csv')
df_2b = pd.read_csv('f_tool2b.csv')
#dl = pickle.load(open('categoryLabels.txt','r'))

from labelling import Classify
classify = Classify(path)
classify.readRE()
y = pd.read_csv('label.csv',names=['filename','labels'])

#assigning column names to the dataframes
#y = pd.DataFrame(dl.items(),columns=['filename', 'Labels'])
df_2a.columns = ['filename','1A','2B','3C','4D','5E','6F','7G','8H','9I','10J',
                 '11K','12L','13M','14N',
                   '15O','16P','17Q','18R','19S','20T']
df_2b.columns = ['filename','A1','B2','C3','D4','E5','F6','G7','H8','I9','J10',
                 'K11','12L','13M']
#y.columns = ['filename','labels']

#formatting the filename column
#df_2a has 2 kinds of patterns in filename column
df_2a.filename = df_2a.filename.str.replace('../../c/',path)
df_2a.filename = df_2a.filename.str.replace('../../../data/c/',path)
df_2b.filename = df_2b.filename.str.replace('../../../data/c/',path)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(df_2a.iloc[:, 1:])
df_2a.iloc[:, 1:] = imputer.transform(df_2a.iloc[:, 1:])
imputer1 = imputer.fit(df_2b.iloc[:, 1:])
df_2b.iloc[:, 1:] = imputer1.transform(df_2b.iloc[:, 1:])

#merging features from 2a, 2b
df_merged = pd.merge(df_2a,df_2b,on='filename',how='inner')

#creating datasets for ml. Merge features with labels
dataset2 = pd.merge(df_merged,y,on='filename',how='inner')
dataset2.to_csv(open('trainXY.csv','w'), sep=' ', index = False)


#feature matrix
dataset2_X = dataset2.iloc[:,1:-1].values

#label vector
dataset2_y = dataset2.iloc[:,-1].values

from ml import Algorithms
algo = Algorithms()

#Automatic Backward Elimination
'''
(m,n) = dataset2_X.shape
dataset2_X = np.append(arr = np.ones((m, 1)).astype(int), values = dataset2_X, axis = 1)
SL = 0.05
X_opt = dataset2_X[:,0:n]
X_Modeled = algo.backwardElimination(X_opt, SL, dataset2_y, m, n)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, dataset2_y, 
                                                    test_size = 0.2)
'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset2_X, dataset2_y, 
                                                    test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Random Forest Algorithm
y_pred = algo.rand_forest(X_train, y_train, X_test)

(m,n) = X_test.shape
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(" Random Forest: {0}".format(accuracy(cm, m)))
