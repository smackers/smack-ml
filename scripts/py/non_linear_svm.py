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
#print files_l
#print len(files_l)
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


'''
# ----- Gaussian kernel
clf = svm.SVC(decision_function_shape = 'ovo')
clf.fit(training_matrix,training_labels)
T = clf.predict(training_matrix)

#---- linear classification
lin_clf = svm.LinearSVC()
lin_clf.fit(training_matrix,training_labels)
S = lin_clf.predict(training_matrix)

#----convert T to a list
T = T.tolist()
S = S.tolist()

#---- calculating accuracy
count1 = 0
count2 = 0
#check for accuracy i.e. correct predictions
for i in range(len(training_labels)):
	if training_labels[i] == T[i]:
		count1 += 1
	if training_labels[i] == S[i]:
		count2 += 1

accuracy1 = str(float(count1*100/len(training_labels))) + '%'
print 'non-linear SVM = {0}'.format(accuracy1)

accuracy2 = str(float(count2*100/len(training_labels))) + '%'
print 'linear SVM = {0}'.format(accuracy2)
'''
