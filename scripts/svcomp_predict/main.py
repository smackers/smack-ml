#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:53:45 2018

@author: ankit
"""

def f1_matrix(y_test, y_pred): #confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = float(sum(np.diag(cm))*100.0/1212)
    return accuracy

def plot_scatter(X_tr, X_te, name): #scatter plots for 2 dimensions (train and test dataset)
    #print(pca.explained_variance_ratio_)
    algo.scatter_plot(X_tr, name, 'train')
    algo.scatter_plot(X_te, name, 'test')

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml import Algorithms
from labelling import Classify

path = '/proj/SMACK/sv-benchmarks/c/'
classify = Classify(path)
classify.readRE()
algo = Algorithms()

#loading data
df_2a = pd.read_csv('f_tool2a.csv') #<tool 1> - features sent by Yulia, SVCOMP 2015
df_2b = pd.read_csv('f_tool2b.csv') #<tool 2> - features generated using Yulia's tool, SVCOMP 2017
#dl = pickle.load(open('categoryLabels.txt','r'))

y = pd.read_csv('label.csv',sep=' ',names=['filename','labels'])

#assigning column names to the dataframes
df_2a.columns = ['filename','1A','2B','3C','4D','5E','6F','7G','8H','9I','10J',
                 '11K','12L','13M','14N',
                   '15O','16P','17Q','18R','19S','20T']
df_2b.columns = ['filename','A1','B2','C3','D4','E5','F6','G7','H8','I9','J10',
                 'K11','12L','13M']

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

df_merged = pd.merge(df_2a,df_2b,on='filename',how='inner') #merging features from 2a, 2b

dataset2 = pd.merge(df_merged,y,on='filename',how='inner') #creating datasets for ml. Merge features with labels
dataset2.to_csv(open('trainXY.csv','w'), sep=' ', index = False) #dumping the features and labels for future loading

dataset2_X = dataset2.iloc[:,1:-1].values #features matrix
dataset2_y = dataset2.iloc[:,-1].values #labels vector

#Automatic Backward Elimination
(m,n) = dataset2_X.shape
#print(m,n)
dataset2_X = np.append(arr = np.ones((m, 1)).astype(int), values = dataset2_X, axis = 1)
SL = 0.05
X_opt = dataset2_X[:,0:n]
X_Modeled = algo.backwardElimination(X_opt, SL, dataset2_y, m, n)

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

print(" Random Forest for original dataset: {0}".format(float(sum(np.diag(cm))*100.0/m)))

cm_df = pd.DataFrame(cm)
#print(cm_df)

'''
Linear Dimensionality reduction to see if Random Forest does a good job.
The data is linearly separable.
'''
print("Choose from: \n 1. PCA \n 2. LDA")
choice = int(input("Enter your choice: "))

if choice == 1:
    # PCA - 90% accuracy
    name = 'PCA'
    for i in range(1,7):
        X_train_m, X_test_m = algo.pca_compute(X_train, X_test, i)
        #if i == 2: plot_scatter(X_train_m, X_test_m, 'PCA')
        y_pred_m = algo.rand_forest(X_train_m, y_train, X_test_m, y_test, name, i)
        print(" Random Forest for PCA with {0} components: {1}".format(i, f1_matrix(y_test,y_pred_m)))

elif choice == 2:
    #LDA - 95% accuracy
    name = 'LDA'
    for i in range(1,7):
        X_train_m, X_test_m = algo.lda_compute(X_train, X_test, y_train, i)
        #if i == 2: plot_scatter(X_train_m, X_test_m, 'LDA')
        y_pred_m = algo.rand_forest(X_train_m, y_train, X_test_m, y_test, name, i)
        print(" Random Forest for LDA with {0} components: {1}".format(i, f1_matrix(y_test,y_pred_m)))

else:
    print("Incorrect option selected")

algo.clustering(X_train, 'Input') #K-Means on actual data
algo.clustering(X_train_m, name) #K-Means on PCA/ LDA dataset
