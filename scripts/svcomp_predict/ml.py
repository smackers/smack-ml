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

    def pca_compute(self, X_train, X_test, i):
        # Applying PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components = i)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        return X_train, X_test

    def lda_compute(self, X_train, X_test, y_train, i):
        #applying LDA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = i)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)

        return X_train, X_test

    def scatter_plot(self, X_train, tmp, tmp2):
        #visualizing the training data
        plt.scatter(X_train[:,0], X_train[:,1], marker='.', c='r')
        plt.title('Visualize ' + tmp + ' results (' + tmp2 + ' data)')
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

    def rand_forest(self, X_train, y_train, X_test, y_test=None, tmp=None, j=0):
        # Fitting Random Forest Classification to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 100, criterion = "entropy", random_state = 0)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        if j == 2:
            self.scatter_plot_dim_red(X_train, y_train, classifier, tmp + ' train set')
            self.scatter_plot_dim_red(X_test, y_test, classifier, tmp + ' test set')
        return y_pred

    def scatter_plot_dim_red(self, X, Y, classifier, name):
        from matplotlib.colors import ListedColormap
        X_set, y_set = X, Y
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue','pink','purple','gray', 'yellow', 'cyan', 'orange', 'lightblue','lightgreen')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', 'blue','pink','purple','gray', 'yellow', 'cyan', 'orange','lightblue','lightgreen'))(i), label = j)
        plt.title(name)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.show()

        #Backward elimination with P-values and adjusted R-square
    def backwardElimination(self, x, SL, y, m ,n):
        import statsmodels.formula.api as sm
        numVars = len(x[0])

        temp = np.zeros((m,n)).astype(int)
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)

            adjR_before = regressor_OLS.rsquared_adj.astype(float)
            if maxVar > SL:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        #print(maxVar, regressor_OLS.pvalues[j].astype(float))
                        temp[:,j] = x[:, j]
                        x = np.delete(x, j, 1)
                        tmp_regressor = sm.OLS(y, x).fit()
                        adjR_after = tmp_regressor.rsquared_adj.astype(float)
                        if (adjR_before >= adjR_after):
                            x_rollback = np.hstack((x, temp[:,[0,j]]))
                            x_rollback = np.delete(x_rollback, j, 1)
                            #print (regressor_OLS.summary())
                            return x_rollback
                        else:
                            continue
        print(regressor_OLS.summary())
        return x
