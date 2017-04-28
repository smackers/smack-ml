from sklearn import svm
import pickle
import numpy as np


A = pickle.load(open('../txt/training/matrix.txt','r'))
y = pickle.load(open('../txt/training/label.txt','r'))
clf = svm.SVC(decision_function_shape = 'ovo')
clf.fit(A,y) 

X = pickle.load(open('../txt/test/matrix.txt','r'))
T = clf.predict(X)

T = T.tolist()

R = pickle.load(open('../txt/test/label.txt','r'))

count = 0

for i in range(len(R)):
	if R[i] == T[i]:
		count = count + 1

accuracy = str(float(count*100/len(R))) + '%'
print accuracy
