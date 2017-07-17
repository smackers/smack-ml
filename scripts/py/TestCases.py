import pickle, random
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Algorithms(object):
	def __init__(self):
		pass
	#-----------------------------------------------------------------------------------------------------
	def RandomSampling(self,matrix_fes):
		N = len(matrix_fes)
		matrix_fes = random.sample(matrix_fes,N)
		return matrix_fes

	def TrainAndTestData(self,matrix_fs,k):
		matrix_fs = self.RandomSampling(matrix_fs)
		#---- picking training & test data
		tr_data = matrix_fs[:k]; te_data = matrix_fs[k:]
		tr_features = []; tr_labels = []
		te_features = []; te_labels = []

		for i in range(len(tr_data)):
			tr_features.append(tr_data[i][:-1])
			tr_labels.append(tr_data[i][-1])

		for i in range(len(te_data)):
			te_features.append(te_data[i][:-1])
			te_labels.append(te_data[i][-1])

		return tr_features, tr_labels, te_features, te_labels

	def ClassificationAlgorithms(self,tr_features,tr_labels,te_features):
		# ----- Gaussian kernel SVM
		clf = svm.SVC(decision_function_shape = 'ovo')
		clf.fit(tr_features,tr_labels)
		T = clf.predict(te_features)

		#---- linear SVM
		lin_clf = svm.LinearSVC()
		lin_clf.fit(tr_features,tr_labels)
		S = lin_clf.predict(te_features)

		# ---- Random Forest ClassificationAlgorithms
		forest_clf = RandomForestClassifier(n_jobs = 2)
		forest_clf.fit(tr_features,tr_labels)
		Z = forest_clf.predict(te_features)

		#----convert T to a list
		T = T.tolist()
		S = S.tolist()
		Z = Z.tolist()

		return T, S, Z

	def ResultAnalysis(self,te_labels, T, S, Z):

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
	def PrintAverage(self,accuracy_non_linear, accuracy_linear, accuracy_forest):
		b1 = reduce(lambda x, y: x+y, accuracy_non_linear)/len(accuracy_non_linear)
		b2 = reduce(lambda x, y: x+y, accuracy_linear)/len(accuracy_linear)
		b3 = reduce(lambda x, y: x+y, accuracy_forest)/len(accuracy_forest)

		print "Accuracy for Non-linear SVM = {0}%%".format(b1)
		print "Accuracy for linear SVM = {0}%%".format(b2)
		print "Accuracy for Random forest classifier = {0}%%".format(b3)

	# ---- print accuracy when # of pass = 1
	def PrintResults(self,a1,a2,a3):
		print "Accuracy for Non-linear SVM = {0}%%".format(a1)
		print "Accuracy for linear SVM = {0}%%".format(a2)
		print "Accuracy for Random forest classifier = {0}%%".format(a3)

	def Just(self,matrix_f,k):
		train_features, train_labels, test_features, test_labels = self.TrainAndTestData(matrix_f,k)
		if len(test_features) == len(test_labels) == 0:
			test_features = train_features
			test_labels = train_labels
		T, S, Z = self.ClassificationAlgorithms(train_features,train_labels,test_features)
		a1, a2,a3 = self.ResultAnalysis(test_labels, T, S, Z)
		return a1,a2,a3


	#-----------------------------------------------------------------------------------------------
	def TestingAlgorithmResults(self,matrix_f):

		print """-----------------Following are the test case options:------------
		1. Run only 1 iteration
		2. Run 'x' # of runs on training data with size 'K'
		3. Run experiment on GAP DEPENDENT values of 'k'
		4. Overfitting
		"""
		t = int(raw_input("Enter the test case you want to run: "))

		if t == 1:
			k = int(raw_input("Enter the number of training samples (<= 7664): "))
			a1,a2,a3 = self.Just(matrix_f,k)
			self.PrintResults(a1,a2,a3)

		elif t == 2:
			k = int(raw_input("Enter the number of training samples (<= 7664): "))
			x = int(raw_input("Enter the number of iterations: "))
			accuracy_non_linear = []
			accuracy_linear = []
			accuracy_forest = []
			for i in range(x):
				a1,a2,a3 = self.Just(matrix_f,k)
				accuracy_non_linear.append(a1)
				accuracy_linear.append(a2)
				accuracy_forest.append(a3)

			self.PrintAverage(accuracy_non_linear,accuracy_linear,accuracy_forest)

		elif t == 3:
			z = int(raw_input("Enter the gap of training set size: "))
			p = []
			for i in range(5500,7501,z):
				p.append(i)

			accuracy_non_linear = []
			accuracy_linear = []
			accuracy_forest = []
			for j in range(len(p)):
				a1,a2,a3 = self.Just(matrix_f,p[j])
				accuracy_non_linear.append(a1)
				accuracy_linear.append(a2)
				accuracy_forest.append(a3)

			self.PrintAverage(accuracy_non_linear,accuracy_linear,accuracy_forest)

		elif t == 4:
			k = len(matrix_f)
			a1,a2,a3 = self.Just(matrix_f,k)
			self.PrintResults(a1,a2,a3)

		else:
			print "Error: Accepted values (1,2,3,4). Check your selected value "
