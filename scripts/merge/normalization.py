#!usr/bin/env python2.7
import numpy as np
import pickle

'''Goal:
1. Perform feature scaling - compute mean
2. Perform feature normalization - compute Standard deviation
'''

def meanStd(fmat):
	mean = []; std = [];
	(m,n) = fmat.shape
	'''last column = <labels>'''
	for i in range(n-1):
		mu = np.mean(fmat[:,i])
		sdev = np.std(fmat[:,i])

		mean.append(mu); std.append(sdev)
		'''perform feature scaling and normalization'''
		if sdev != 0: fmat[:,i] = (fmat[:,i] - mu)/sdev;
		else: fmat[:,i] = fmat[:,i] - mu;
	return fmat, mean, std

if __name__ == '__main__':
	mat = pickle.load(open('notNormalizedMergedLabels.txt','r'))
	#(m,n) = mat.shape
	#print mat[0,:]
	norm_mat, mean, std = meanStd(mat)
	#print norm_mat[0,:]
	with open('normalizedMergedLabels.txt','w') as f:
		pickle.dump(norm_mat,f)
