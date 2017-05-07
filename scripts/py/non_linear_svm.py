from sklearn import svm
import pickle
import numpy as np
import pickle

files_f = pickle.load(open('../txt/list_of_features.txt','rb'))
files_l = pickle.load(open('../txt/final_labels.txt','rb'))
vector_l = []
matrix_f = []
k = int(raw_input("Enter the number of training samples (<= 260): "))

'''		merging the feature vectors with their respective labels 	'''

for f in files_l:
	if f in files_f:
		matrix_f.append(files_f[f])
		vector_l.append(files_l[f])


training_matrix = matrix_f[:k]
training_labels = vector_l[:k]
test_matrix = matrix_f[k:]
test_labels = vector_l[k:]

clf = svm.SVC(decision_function_shape = 'ovo')
clf.fit(training_matrix,training_labels) 

T = clf.predict(test_matrix)

#convert T to a list
T = T.tolist()

count = 0

#check for accuracy i.e. correct predictions
for i in range(len(test_labels)):
	if test_labels[i] == T[i]:
		count = count + 1

accuracy = str(float(count*100/len(test_labels))) + '%'
print accuracy
