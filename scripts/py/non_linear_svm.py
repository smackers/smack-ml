from sklearn import svm
import pickle
import numpy as np
import pickle
import random

'''
	Purpose: Training the model and testing
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

k = int(raw_input("Enter the number of training samples (<= 7671): "))

# ---- random sampling of training data
tr_data = random.sample(matrix_f,k)
tr_features = []
tr_labels = []

# ---- separating training features and labels after sampling
for i in range(len(tr_data)):
	tr_features.append(tr_data[i][:-1])
	tr_labels.append(tr_data[i][-1])

J = int(raw_input("Enter the number of test samples (<= {0}): ".format(k)))

# ---- random sampling of test data
te_data = random.sample(matrix_f,J)
te_features = []
te_labels = []

# ---- separating training features and labels after sampling
for i in range(len(te_data)):
	te_features.append(te_data[i][:-1])
	te_labels.append(te_data[i][-1])


# ----- Gaussian kernel
clf = svm.SVC(decision_function_shape = 'ovo')
clf.fit(tr_features,tr_labels)
T = clf.predict(te_features)

#---- linear classification
lin_clf = svm.LinearSVC()
lin_clf.fit(tr_features,tr_labels)
S = lin_clf.predict(te_features)

#----convert T to a list
T = T.tolist()
S = S.tolist()

#---- calculating accuracy
count1 = 0
count2 = 0
#check for accuracy i.e. correct predictions
for i in range(len(te_labels)):
	if te_labels[i] == T[i]:
		count1 += 1
	if te_labels[i] == S[i]:
		count2 += 1

accuracy1 = (float(count1)/len(te_labels))*100
print "non-linear SVM = %.4f %%" %accuracy1

accuracy2 = (float(count2)/len(te_labels))*100
print "linear SVM = %.4f %%" %accuracy2
