from sklearn import svm
import pickle


A = pickle.load(open('../txt/training/matrix.txt','r')
y = pickle.load(open('../txt/training/label.txt','r')
clf = svm.SVC()
clf.fit(A,y) 
