import pickle, random
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

	#print tr_labels
	print te_labels

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

	# ---- Random Forest classification
	forest_clf = RandomForestClassifier(n_jobs = 2)
	forest_clf.fit(tr_features,tr_labels)
	Z = forest_clf.predict(te_features)

	#----convert T to a list
	T = T.tolist()
	S = S.tolist()
	Z = Z.tolist()

	return T, S, Z

def analyzing(te_labels, T, S, Z):
	
	#---- calculating accuracy
	count1 = 0.0
	count2 = 0.0
	count3 = 0.0
	#check for accuracy i.e. correct predictions
	for i in range(len(te_labels)):
		if te_labels[i] == T[i]:
			count1 += 1
		if te_labels[i] == S[i]:
			count2 += 1
		if te_labels[i] == Z[i]:
			count3 += 1

	accuracy1 = count1/len(te_labels)*100
	accuracy2 = count2/len(te_labels)*100
	accuracy3 = count3/len(te_labels)*100

	return accuracy1, accuracy2, accuracy3

# ---- print average accuracy when # of pass = x 
def print_average(accuracy_non_linear, accuracy_linear, accuracy_forest):
	b1 = reduce(lambda x, y: x+y, accuracy_non_linear)/len(accuracy_non_linear)
	b2 = reduce(lambda x, y: x+y, accuracy_linear)/len(accuracy_linear)
	b3 = reduce(lambda x, y: x+y, accuracy_forest)/len(accuracy_forest)

	print "Accuracy for Non-linear SVM = {0}%%".format(b1)
	print "Accuracy for linear SVM = {0}%%".format(b2)
	print "Accuracy for Random forest classifier = {0}%%".format(b3)

# ---- print accuracy when # of pass = 1
def print_results(a1,a2,a3):
	print "Accuracy for Non-linear SVM = {0}%%".format(a1)
	print "Accuracy for linear SVM = {0}%%".format(a2)
	print "Accuracy for Random forest classifier = {0}%%".format(a3)

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
	T, S, Z = classification(tr_f,tr_l,te_f)
	a1, a2,a3 = analyzing(te_l, T, S, Z)
	print_results(a1,a2,a3)

elif t == 2:
	k = int(raw_input("Enter the number of training samples (<= 7671): "))
	x = int(raw_input("Enter the number of iterations: "))
	accuracy_non_linear = []
	accuracy_linear = []
	accuracy_forest = []
	for i in range(x):
		tr_f, tr_l, te_f, te_l = picking_data(matrix_f,k)
		T, S, Z = classification(tr_f,tr_l,te_f)
		a1, a2,a3 = analyzing(te_l, T, S, Z)
		accuracy_non_linear.append(a1)
		accuracy_linear.append(a2)
		accuracy_forest.append(A3)

	print_average(accuracy_non_linear,accuracy_linear,accuracy_forest)

elif t == 3:
	z = int(raw_input("Enter the gap of training set size: "))
	p = []
	for i in range(4000,7501,z):
		p.append(i)
	
	accuracy_non_linear = []
	accuracy_linear = []
	accuracy_forest = []
	for j in range(len(p)):
		tr_f, tr_l, te_f, te_l = picking_data(matrix_f,p[j])
		T, S, Z = classification(tr_f,tr_l,te_f)
		a1, a2, a3 = analyzing(te_l, T, S, Z)
		accuracy_non_linear.append(a1)
		accuracy_linear.append(a2)
		accuracy_forest.append(a3)
	
	print_average(accuracy_non_linear,accuracy_linear,accuracy_forest)

elif t == 4:
	k = len(matrix_f)
	tr_f, tr_l, te_f, te_l = picking_data(matrix_f,k)
	te_f = tr_f
	te_l = tr_l
	T, S, Z = classification(tr_f,tr_l,te_f)
	a1, a2, a3 = analyzing(te_l, T, S, Z)
	print_results(a1,a2,a3)

else:
	print "Error: Accepted values (1,2,3,4). Check your selected value "
