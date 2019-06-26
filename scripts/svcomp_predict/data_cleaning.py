
# coding: utf-8

# In[5]:


#!/usr/bin/env python3


# In[6]:


class preprocessing(object):
        def __init__(self):
            pass

        def load_data(self, path = '/proj/SMACK/sv-benchmarks/c/'):
            import pandas as pd
            #<tool 1> - features sent by Yulia, SVCOMP 2015
            X_2a = pd.read_csv('f_tool2a.csv', header='infer')
            #<tool 2> - features generated using Yulia's tool, SVCOMP 2017
            X_2b = pd.read_csv('f_tool2b.csv')

            """ X_2a.columns = ['filename','alloc_size','array_index','bitvector',
            'boolean','branc_cond','char_def','char_use','const_assign','counter',
            'file_def','file_use','input','linear','loop_bound','loop_iterator',
            'mode','offset','synt_const','unresolved_assign','used_in_arithm']
            """
            X_2b.columns = ['filename','A1','B2','C3','D4','E5','F6','G7','H8','I9','J10','K11','L12','M13']

            #formatting the filename column
            #df_2a has 2 kinds of patterns in filename column
            X_2a.filename = X_2a.filename.str.replace('../../c/',path)
            X_2a.filename = X_2a.filename.str.replace('../../../data/c/',path)

            X_2b.filename = X_2b.filename.str.replace('../../../data/c/',path)

            y = pd.read_csv('label.csv',sep=' ',names=['filename','labels'])

            #merging features from 2a, 2b
            X_merged = pd.merge(X_2a,X_2b,on='filename',how='inner')

            #creating datasets for ml. Merge features with labels
            dataset = pd.merge(X_merged,y,on='filename',how='inner')
            return dataset

        def missing_data(self, dataset):
            from sklearn.impute import SimpleImputer
            import numpy as np

            imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
            imputer = imputer.fit(dataset.iloc[:, 1:])
            dataset.iloc[:, 1:] = imputer.transform(dataset.iloc[:, 1:])
            return dataset

        def feature_scaling(self, X_train, X_test = []):
            from sklearn.preprocessing import StandardScaler
            #standardization
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            if len(X_test) != 0: X_test = sc.transform(X_test);
            return X_train, X_test

'''if __name__=='__main__':
    prep = preprocessing()
    dataset = prep.load_data()
    print(dataset.columns)
'''
