import pickle, random
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Algorithms(object):
	def __init__(self, train_X, train_y, test_X, test_y):
		train_X = self.train_X
        train_y = self.train_y
        test_X = self.test_X
        test_y = self.test_y

	def ClassificationAlgorithms(self):
		# ----- Gaussian kernel SVM
		clf = svm.SVC(decision_function_shape = 'ovo')
		clf.fit(self.train_X,self.train_y)
		T = clf.predict(self.test_X)

		#---- linear SVM
		lin_clf = svm.LinearSVC()
		lin_clf.fit(self.train_X,self.train_y)
		S = lin_clf.predict(self.test_X)

		# ---- Random Forest ClassificationAlgorithms
		forest_clf = RandomForestClassifier(n_jobs = 2)
		forest_clf.fit(self.train_X,self.train_y)
		Z = forest_clf.predict(self.test_X)

		#----convert T to a list
		T = T.tolist()
		S = S.tolist()
		Z = Z.tolist()

		return T, S, Z

	def ResultAnalysis(self,self.test_y, T, S, Z):
		count1 = 0.0; count2 = 0.0; count3 = 0.0
		#---- calculating accuracy
		for i in range(len(self.test_y)):
			if self.test_y[i] == T[i]:
				count1 += 1
			if self.test_y[i] == S[i]:
				count2 += 1
			if self.test_y[i] == Z[i]:
				count3 += 1
		#check for accuracy i.e. correct predictions
		accuracy1 = count1/len(self.test_y)*100
		accuracy2 = count2/len(self.test_y)*100
		accuracy3 = count3/len(self.test_y)*100

		return accuracy1, accuracy2, accuracy3

	def AlgorithmTesting(self):
		T, S, Z = self.ClassificationAlgorithms()
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