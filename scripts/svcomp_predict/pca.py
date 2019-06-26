#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:44:57 2019

@author: ankit
"""
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ml_models import supervised, unsupervised, dim_red
from data_cleaning import preprocessing
#from visualization import plots
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 10)
plt.style.use('ggplot')



def sc_plot_with_label(df):
    sns.set()
    sns.scatterplot(data=df, x ='x1', y ='x2',hue='labels',legend='full',
                       palette = sns.color_palette('hls',11), alpha=1)
    plt.show()
        
def heatmaps(df):
    df_ = pd.DataFrame(df, columns=None)
    corr_ = df_.corr()
    ax = sns.heatmap(corr_, vmin=-1, vmax=1, center=0,
                     cmap=sns.diverging_palette(20, 220, n=200),square=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')

def top_two_components(df):
    #dataset.to_csv('dataset_pca.csv')
    top_two = pd.DataFrame(columns=['x1','x2'])
    top_two['x1'] = df[:,0]
    top_two['x2'] = df[:,1]
    top_two['labels'] = y
    
    sc_plot_with_label(top_two)

def tsne_plots(df):
    """
    1. Apply t-SNE on LDA (9 dims) and visualize the results. t-SNE is able to capture
    non-linear relationships between features while performing dim reduction so 
    expected results should be better than PCA (although t-SNE is very slow)
    2. PCA is mathematical technique, t-SNE is probabilistic
    3. Visualizing t-SNE in 2D.
    """
    dataset_tsne=TSNE(n_iter=300, n_components=2).fit_transform(df)
    tsne_top_two = pd.DataFrame(columns=['x1','x2'])
    tsne_top_two['x1'] = dataset_tsne[:,0]
    tsne_top_two['x2'] = dataset_tsne[:,1]
    tsne_top_two['labels'] = y
    #print(tsne_top_two['labels'].nunique())
    sc_plot_with_label(tsne_top_two)
    return dataset_tsne

def plot_curves(new_dataset, status, y=[]):
    retention = []
    dimensions = [i for i in range(1, len(new_dataset.columns)-1)]
    for i in range(1,len(new_dataset.columns)-1):
        if status == 'PCA':
            X_train, X_test, variance_ratio = obj.pca_compute(i)
        if status == 'LDA':
            X_train, X_test, variance_ratio = obj.lda_compute(i,y)
        retention.append(sum(variance_ratio))
    
    plt.plot(dimensions, retention)
    plt.title(status)
    plt.xlabel('# of dimensions')
    plt.ylabel('variance ratio')
    plt.show()
    
"""
preprocessing of the original dataset
1. Loading features from both tools and merging them - load_data()
2. Replacing any missing data with the 'mean' of the feature values - missing_data()
3. Min-Max ormalizing the data - MinMaxScaler()
"""
scaler = MinMaxScaler()
dataset = preprocessing().load_data()
dataset = preprocessing().missing_data(dataset)
dataset.to_csv('dataset.csv')
new_dataset = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1].values #labels vector
new_dataset[new_dataset.columns] = scaler.fit_transform(new_dataset[new_dataset.columns])

obj = dim_red(new_dataset)
"""
Correlation matrix for the original dataset
"""
heatmaps(new_dataset)
"""
Observing behavior of PCA for different # of output features
"""
plot_curves(new_dataset,'PCA')

"""
1. Achieve 99.4% retention/ variance ratio at dims = 21. Perform
dim reduction to 21 without loosing significant information or performance of the ML
algorithm in later stages.
2. Resulting features have no correlation with each other.
3. visualizing PCA in 2D for 21 dims input by selecting the 
top 2 principal components to.
"""
dataset_pca, X_test, variance_ratio = obj.pca_compute(21)
print(sum(variance_ratio))
heatmaps(dataset_pca)

top_two_components(dataset_pca)
pca_tsne_df = tsne_plots(dataset_pca)


plot_curves(new_dataset,'LDA', y)

"""
1. Achieve 99.7% retention/ variance ratio at dims = 9. Perform
dim reduction to 21 without loosing significant information or performance of the ML
algorithm in later stages.
2. Resulting features have no correlation with each other.
3. visualizing LDA in 2D for 9 dims input by selecting the 
top 2 components.
"""
dataset_lda, X_test, variance_ratio = obj.lda_compute(9,y)
print(sum(variance_ratio))
heatmaps(dataset_lda)
top_two_components(dataset_lda)
lda_tsne_df = tsne_plots(dataset_lda)