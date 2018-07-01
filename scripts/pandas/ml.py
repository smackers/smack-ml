#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:50:05 2018

@author: ankit
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Algorithms():
    def __init__(self):
        pass
    
    def pca_compute(self, X_train, X_test):
        # Applying PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(pca.explained_variance_ratio_)
        
        self.scatter_plot(X_train, 'PCA')
        return X_train, X_test
        
    def lda_compute(self, X_train, X_test, y_train):
        #applying LDA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = 2)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
        
        self.scatter_plot(X_train, 'LDA')
        return X_train, X_test
    
    def kpca_compute(self, X_train, X_test):
        # Applying Kernel PCA
        from sklearn.decomposition import KernelPCA
        kpca = KernelPCA(n_components=2, kernel='rbf')
        X_train = kpca.fit_transform(X_train)
        X_test = kpca.transform(X_test)
        
        #self.scatter_plot(X_train, 'K-PCA')
        return X_train, X_test
        
    def scatter_plot(self, X_train, tmp):
        #visualizing the training data
        plt.scatter(X_train[:,0], X_train[:,1], marker='.', c='r')
        plt.title('Visualize ' + tmp + ' results (Training data)')        
        plt.xlabel(tmp + '1')
        plt.ylabel(tmp + '2')
        plt.show()
    
    def clustering(self, X):
        #Applying K-Means clustering
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(1,12):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init = 10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            
        #visualizing the WCSS curve
        plt.plot(range(1,12), wcss)    
        plt.title('The Elbow method (Training Data)')
        plt.xlabel('# of clusters')
        plt.ylabel('WCSS')           
        plt.show()


    def logistic_reg(self, X_train, y_train, X_test):
        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        pred = classifier.predict(X_test)
        return pred

    def rand_forest(self, X_train, y_train, X_test):
        # Fitting Random Forest Classification to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        pred = classifier.predict(X_test)
        return pred