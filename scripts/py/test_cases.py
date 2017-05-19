import pickle, random
from sklearn import svm
import numpy as np

#-----------------------------------------------------------------------------------------------------
def random_sampling(matrix_fes):
	N = len(matrix_fes)
	matrix_fes = random.sample(matrix_fes,N)
	return matrix_fes

def picking_data(matrix_fs,k):
	matrix_fs = random_sampling(matrix_fs)
	#---- picking training & test data
	tr_data = matrix_f[:k]; te_data = matrix_f[k:]
	tr_features = []; tr_labels = []
	te_features = []; te_labels = []
	
	for i in range(len(tr_data)):
		tr_features.append(tr_data[i][:-1])
		tr_labels.append(tr_data[i][-1])

	for i in range(len(te_data)):
		te_features.append(te_data[i][:-1])
		te_labels.append(te_data[i][-1])

	return tr_features, tr_labels, te_features, te_labels

def classification(tr_features,tr_labels,te_features):
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

	return T, S

def analyzing(te_labels, T, S):
	
	#---- calculating accuracy
	count1 = 0.0
	count2 = 0.0
	#check for accuracy i.e. correct predictions
	for i in range(len(te_labels)):
		if te_labels[i] == T[i]:
			count1 += 1
		if te_labels[i] == S[i]:
			count2 += 1

	accuracy1 = count1/len(te_labels)*100
	accuracy2 = count2/len(te_labels)*100
	
	return accuracy1, accuracy2

#-----------------------------------------------------------------------------------------------
matrix_f = pickle.load(open("../txt/f_matrix.txt","rb"))

print """-----------------Following are the test case options:------------
1. Run only 1 iteration
2. Run 'x' # of runs on training data with size 'K'
3. Run experiment on GAP DEPENDENT values of 'k'
4. Overfitting
"""
t = int(raw_input("Enter the test case you want to run: "))

if t == 1:
	k = int(raw_input("Enter the number of training samples (<= 7671): "))
	tr_f, tr_l, te_f, te_l = picking_data(matrix_f,k)
	T, S = classification(tr_f,tr_l,te_f)
	a1, a2 = analyzing(te_l, T, S)
	print "Accuracy for Non-linear SVM = {0}%%".format(a1)
	print "Accuracy for linear SVM = {0}%%".format(a2)

elif t == 2:
	k = int(raw_input("Enter the number of training samples (<= 7671): "))
	x = int(raw_input("Enter the number of iterations: "))
	accuracy_non_linear = []
	accuracy_linear = []
	for i in range(x):
		tr_f, tr_l, te_f, te_l = picking_data(matrix_f,k)
		T, S = classification(tr_f,tr_l,te_f)
		a1, a2 = analyzing(te_l, T, S)
		accuracy_non_linear.append(a1)
		accuracy_linear.append(a2)

	b1 = reduce(lambda x, y: x+y, accuracy_non_linear)/len(accuracy_non_linear)
	b2 = reduce(lambda x, y: x+y, accuracy_linear)/len(accuracy_linear)
	print "Accuracy for Non-linear SVM = {0}%%".format(b1)
	print "Accuracy for linear SVM = {0}%%".format(b2)

elif t == 3:
	z = int(raw_input("Enter the gap of training set size: "))
	p = []
	for i in range(4000,7501,z):
		p.append(i)
	
	accuracy_non_linear = []
	accuracy_linear = []
	for j in range(len(p)):
		tr_f, tr_l, te_f, te_l = picking_data(matrix_f,p[j])
		T, S = classification(tr_f,tr_l,te_f)
		a1, a2 = analyzing(te_l, T, S)
		accuracy_non_linear.append(a1)
		accuracy_linear.append(a2)
	
	b1 = reduce(lambda x, y: x+y, accuracy_non_linear)/len(accuracy_non_linear)
	b2 = reduce(lambda x, y: x+y, accuracy_linear)/len(accuracy_linear)
	print "Accuracy for Non-linear SVM = {0}%%".format(b1)
	print "Accuracy for linear SVM = {0}%%".format(b2)

elif t == 4:
	k = len(matrix_f)
	tr_f, tr_l, te_f, te_l = picking_data(matrix_f,k)
	te_f = tr_f
	te_l = tr_l
	T, S = classification(tr_f,tr_l,te_f)
	a1, a2 = analyzing(te_l, T, S)
	print "Accuracy for Non-linear SVM = {0}%%".format(a1)
	print "Accuracy for linear SVM = {0}%%".format(a2)

else:
	print "Error: Accepted values (1,2,3). Check your selected value "
