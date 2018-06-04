import glob2, pickle
import numpy as np
from TestCases import Algorithms

from setFiles import classify
from normalization import normalize

pathname = '/proj/SMACK/smack-ml/scripts/txt/'
a1 = classify()
labels = a1.readRE()

#print labels, print len(labels)

f = open(pathname + 'FinalFeatures.txt','r')
features = pickle.load(f)
#print len(features), len(features[0]), features

final = {}; path = '/proj/SMACK/sv-benchmarks/'; mat = []
for item in features:
	tmp = path+item[6:]
	if tmp in labels:
		if len(features[item]) == 33:
			final[tmp] = features[item] + [labels[tmp]]
			mat.append(final[tmp])
#print final, print matrix

print len(mat)
#np_array = np.empty([len(matrix),len(matrix[0])])
#np_matrix = np.matrix([xi for xi in matrix])
#print np_array
np_mat = np.matrix(mat)
(m,n) = np.shape(np_mat)
#print np_matrix

a3 = normalize(np_mat, m , n)
norm_mat, mean, std = a3.meanNorm()
#print norm_mat

a2 = Algorithms()
a2.TestCasesForAlgorithm(norm_mat,0)
