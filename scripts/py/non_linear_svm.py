from sklearn import svm
import pickle
import numpy as np
import pickle
import random

'''
	Purpose: creating the main matrix with features & labels
'''

files_f = pickle.load(open('../txt/list_of_features.txt','rb'))
files_l = pickle.load(open('../txt/final_labels.txt','rb'))

vector_l = []
matrix_f = []

# ---- matrix_f = [feature_vector, label]
for f in files_l:
	if f in files_f:
		files_f[f].append(files_l[f])
		matrix_f.append(files_f[f])

with open("../txt/f_matrix.txt","w") as f:
	pickle.dump(matrix_f,f)
