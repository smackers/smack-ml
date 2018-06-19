import glob2, pickle
import numpy as np
from sklearn.cluster import KMeans

from TestCases import Algorithms
from setFiles import classify
from normalization import normalize
from analyzing import analysis
from analyze_mat import ana

pathname = '/proj/SMACK/smack-ml/scripts/txt/'
a1 = classify()
labels = a1.readRE()

with open('categoryLabels.txt','w') as f:
	pickle.dump(labels,f)
#print labels, print len(labels)

f = open(pathname + 'FinalFeatures.txt','r')
features = pickle.load(f)
f.close()
#print len(features), len(features[0]), features

final = {}; path = '/proj/SMACK/sv-benchmarks/'; mat = []
for item in features:
	tmp = path+item[6:]
	if tmp in labels:
		if len(features[item]) == 33:
			final[tmp] = features[item] + [labels[tmp]]
			mat.append(final[tmp])


with open('NormFeaturesCategoryLabels.txt','w') as f:
	pickle.dump(mat,f)
f.close()

np_mat = np.matrix(mat)
#print np_mat
(m,n) = np.shape(np_mat)
#print m,n

a3 = normalize(np_mat, m , n)
norm_mat, mean, std = a3.meanStd()

with open('normalizedF.txt','w') as f:
	pickle.dump(norm_mat,f)

cluster_mat = norm_mat[:,0:n-1]
kmeans = KMeans(n_clusters=11, random_state=0).fit(cluster_mat)
cl = kmeans.labels_
print type(kmeans.cluster_centers_)

'''
#print kmeans
a9 = analysis(kmeans, cluster_mat)
a9.clustering(cl)
'''
'''

norm_mat1 = np.matrix('1,0,2;0,1,2;3,2,1')
a11 = ana(norm_mat1,cl)
a11.mat_clus()
'''
'''
a2 = Algorithms()
a2.TestCasesForAlgorithm(norm_mat,0)
'''
