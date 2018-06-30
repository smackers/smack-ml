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
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def pca_compute(self):
        # Applying PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 2)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        #explained_variance = pca.explained_variance_ratio_
        final_pred, classifier = self.rand_forest()
        self.visualize_data()
        #self.visualize('Random Forest', classifier)
        return final_pred
        
    def logistic_reg(self):
        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        
        # Predicting the Test set results
        pred = classifier.predict(self.X_test)
        return pred

    def rand_forest(self):
        # Fitting Random Forest Classification to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
        classifier.fit(self.X_train,self.y_train)
        
        # Predicting the Test set results
        pred = classifier.predict(self.X_test)
        return pred, classifier
    
    def visualize_data(self):
        plt.scatter(self.X_test[:,0],self.X_test[:,1], marker='.', c='r')
        plt.title('Visualize PCA results')        
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.show()
        
        
    def visualize_results(self, string, classifier):
        
        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = self.X_train, self.y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('#ff0000', '#ff8000', '#bfff00','#40ff00',
                                            '#00ffff', '#0080ff', '#bf00ff', '#ff00ff',
                                            '#808080', '#ffe6e6', '#330000'))(i), label = j)
        plt.title(string + ' (Training set)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.show()
        
        # Visualising the Test set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = self.X_test, self.y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('#ff0000', '#ff8000', '#bfff00','#40ff00',
                                            '#00ffff', '#0080ff', '#bf00ff', '#ff00ff',
                                            '#808080', '#ffe6e6', '#330000'))(i), label = j)
        plt.title(string + ' (Test set)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.show()

        