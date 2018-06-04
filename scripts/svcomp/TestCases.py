import pickle, random
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Algorithms(object):
	def __init__(self):
		pass
	#-----------------------------------------------------------------------------------------------------

	def partitionData(self,tr_mat,te_mat):
		tr_f = tr_mat[:,0:-1]
		tr_l = np.ravel(tr_mat[:,-1])
		te_f = te_mat[:,0:-1]
		te_l = np.ravel(te_mat[:,-1])
		return tr_f, tr_l, te_f, te_l


	def ClassificationAlgorithms(self,tr_f,tr_l,te_f):
		# ----- Gaussian kernel SVM
		clf = svm.SVC(decision_function_shape = 'ovo')
		clf.fit(tr_f,tr_l)
		T = clf.predict(te_f)

		#---- linear SVM
		lin_clf = svm.LinearSVC()
		lin_clf.fit(tr_f,tr_l)
		S = lin_clf.predict(te_f)

		# ---- Random Forest ClassificationAlgorithms
		forest_clf = RandomForestClassifier(n_jobs = 2)
		forest_clf.fit(tr_f,tr_l)
		Z = forest_clf.predict(te_f)

		#----convert T to a list
		T = T.tolist()
		S = S.tolist()
		Z = Z.tolist()

		return T, S, Z

	def ResultAnalysis(self,te_l, T, S, Z):
		count1 = 0.0; count2 = 0.0; count3 = 0.0
		#---- calculating accuracy
		for i in range(len(te_l)):
			if te_l[i] == T[i]:
				count1 += 1
			if te_l[i] == S[i]:
				count2 += 1
			if te_l[i] == Z[i]:
				count3 += 1
		#check for accuracy i.e. correct predictions
		accuracy1 = count1/len(te_l)*100
		accuracy2 = count2/len(te_l)*100
		accuracy3 = count3/len(te_l)*100

		return accuracy1, accuracy2, accuracy3

	def AlgorithmTesting(self,matrix_f,k):
		N = len(matrix_f)
		np.random.shuffle(matrix_f)
		#---- picking training & test data
		tr_data = matrix_f[:k,:]; te_data = matrix_f[k:,:]
		print tr_data
		train_features, train_labels, test_features, test_labels = self.partitionData(tr_data, te_data)
	
		if len(test_features) == len(test_labels) == 0:	#overfitting case
			test_features = train_features
			test_labels = train_labels
		
		T, S, Z = self.ClassificationAlgorithms(train_features,train_labels,test_features)
		a1, a2,a3 = self.ResultAnalysis(test_labels, T, S, Z)
		return a1,a2,a3

	# ---- compute average accuracy when # of pass > 1
	def PrintAverage(self,accuracy_non_linear, accuracy_linear, accuracy_forest):
		b1 = reduce(lambda x, y: x+y, accuracy_non_linear)/len(accuracy_non_linear)
		b2 = reduce(lambda x, y: x+y, accuracy_linear)/len(accuracy_linear)
		b3 = reduce(lambda x, y: x+y, accuracy_forest)/len(accuracy_forest)

		self.PrintResults(b1,b2,b3)

	# ---- print accuracy
	def PrintResults(self,a1,a2,a3):
		print "Accuracy for Non-linear SVM = {0}%%".format(a1)
		print "Accuracy for linear SVM = {0}%%".format(a2)
		print "Accuracy for Random forest classifier = {0}%%".format(a3)
	#-----------------------------------------------------------------------------------------------
	def TestCasesForAlgorithm(self,matrix_f,matrix_newData):

		print """-----------------Following are the test case options:------------
		1. Run only 1 iteration
		2. Run 'x' # of runs on training data with size 'K'
		3. Run experiment on GAP DEPENDENT values of 'k'
		4. Overfitting
		5. Predict labels for new data"""
		t = int(raw_input("Enter the test case you want to run: "))

		if t == 1:
			k = int(raw_input("Enter the number of training samples (<= 7664): "))
			a1,a2,a3 = self.AlgorithmTesting(matrix_f,k)
			self.PrintResults(a1,a2,a3)

		elif t == 2:
			k = int(raw_input("Enter the number of training samples (<= 7664): "))
			x = int(raw_input("Enter the number of iterations: "))
			accuracy_non_linear = []
			accuracy_linear = []
			accuracy_forest = []
			for i in range(x):
				a1,a2,a3 = self.AlgorithmTesting(matrix_f,k)
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
				a1,a2,a3 = self.AlgorithmTesting(matrix_f,p[j])
				accuracy_non_linear.append(a1)
				accuracy_linear.append(a2)
				accuracy_forest.append(a3)

			self.PrintAverage(accuracy_non_linear,accuracy_linear,accuracy_forest)

		elif t == 4:
			k = len(matrix_f)
			a1,a2,a3 = self.AlgorithmTesting(matrix_f,k)
			self.PrintResults(a1,a2,a3)

		elif t == 5:
			tr_data, tr_label = self.TrainingData(matrix_f)
			NLSVM, LSVM, RF = self.ClassificationAlgorithms(tr_data, tr_label, matrix_newData)

		else:
			print "Error: Accepted values (1,2,3,4). Check your selected value "
