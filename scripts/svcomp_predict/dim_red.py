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
import matplotlib.pyplot as plt

path = '/proj/SMACK/sv-benchmarks/c/'

dataset2 = pd.read_csv('trainXY.csv', sep=' ')

#feature matrix
dataset2_X = dataset2.iloc[:,1:-1].values

#label vector
dataset2_y = dataset2.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset2_X, dataset2_y, 
                                                    test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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
