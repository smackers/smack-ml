#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:53:45 2018

@author: ankit
"""
def f1_matrix(y_test, y_pred): #confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    accuracy = sum(np.diag(cm))/1212
    #print(accuracy)
    return accuracy*100

#importing libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

path = '/proj/SMACK/sv-benchmarks/c/'

data = pd.read_csv('featureVec.txt',sep=' ',header = None)
data.columns = ['filename','Aliasing','Arrays','Boolean','Composite']

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer2 = imputer.fit(data.iloc[:,1:])
data.iloc[:,1:] = imputer2.transform(data.iloc[:,1:])

# Generating labels
from labelling import Classify
classify = Classify(path)
classify.readRE()
y = pd.read_csv('label.txt',sep=' ', header = None)
y.columns = ['filename', 'labels']

data = pd.merge(data,y,left_on='filename',right_on='filename',how='inner')

#print(len(y),len(data.labels))
#print(len(y[y.labels == 1]), len(data[data.labels == 1]))
#feature matrix
data_X = data.iloc[:,1:-1].values

#label vector
data_y = data.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, 
                                                    test_size = 0.2)

#print(len(X_train), len(X_test))

from ml import Algorithms
algo = Algorithms()

'''
Linear Dimensionality reduction to see if Random Forest does a good job.
The data is linearly separable.
'''

print("Choose from: \n 1. PCA \n 2. LDA \n 3. Kernel-PCA")
choice = int(input("Enter your choice: "))

if choice == 1:
    # PCA - 90% accuracy
    X_train, X_test = algo.pca_compute(X_train, X_test)

elif choice == 2:
    #LDA - 95% accuracy
    X_train, X_test = algo.lda_compute(X_train, X_test, y_train)

elif choice == 3:
    #Kernel-PCA (non-linear Dimensionality Reduction) - 63% accuracy
    X_train, X_test = algo.kpca_compute(X_train, X_test)

else:
    print("Incorrect option selected")
    
#K-Means
algo.clustering(X_train)

#Random Forest algorithm
y_pred = algo.rand_forest(X_train, y_train, X_test)
print(" Random Forest: {0}".format(f1_matrix(y_test,y_pred)))