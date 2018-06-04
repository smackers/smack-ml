import glob2, pickle
import numpy as np
from TestCases import Algorithms

from setFiles import classify

pathname = '/proj/SMACK/smack-ml/scripts/txt/'
a1 = classify()
labels = a1.readRE()

#print labels, print len(labels)

f = open(pathname + 'FinalFeatures.txt','r')
features = pickle.load(f)

#print len(features), len(features[0]), features

final = {}; path = '/proj/SMACK/sv-benchmarks/'; matrix = []
for item in features:
	tmp = path+item[6:]
	if tmp in labels:
		if len(features[item]) == 33:
			final[tmp] = features[item] + [labels[tmp]]
			matrix.append(final[tmp])
#print final, print matrix

print len(matrix)
#np_array = np.empty([len(matrix),len(matrix[0])])
np_array = np.array([xi for xi in matrix])
#print np.shape(np_array)
#print np_array
np_matrix = np.array(matrix)

a2 = Algorithms()
a2.TestCasesForAlgorithm(matrix,0)
