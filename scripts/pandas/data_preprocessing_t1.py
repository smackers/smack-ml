#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:53:45 2018

@author: ankit
"""

#importing libraries
import numpy as np
import pandas as pd
import pickle

path = '/proj/SMACK/sv-benchmarks/c/'

#<tool 1> - features sent by Yulia, SVCOMP 2015
#<tool 2> - features generated using Yulia's tool, SVCOMP 2017
#loading data
df1 = pd.read_csv('f_tool1.csv')
dl = pickle.load(open('categoryLabels.txt','r'))

#assigning column names to the dataframes
y = pd.DataFrame(dl.items(),columns=['filename', 'Labels'])

#formatting the filename column
#df_2a has 2 kinds of patterns in filename column
df1.file = df1.file.str.replace('/home/vagrant/benchmarks/svcomp15/',path)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer2 = imputer.fit(df1.iloc[:,1:])
df1.iloc[:,1:] = imputer2.transform(df1.iloc[:,1:])

#creating datasets for ml. Merge features with labels
dataset1 = pd.merge(df1,y,left_on='file',right_on='filename',how='inner')

#feature matrix
dataset1_X = dataset1.iloc[:,:-1].values

#label vector
dataset1_y = dataset1.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(dataset1_X, dataset1_y, test_size = 0.2, random_state = 0)

