import numpy as np
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt

norm_mat = pickle.load(open('normalizedMergedLabels.txt','r'))

'''
pca = PCA().fit(norm_mat)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('# of components')
plt.ylabel('cumulative explained_variance_')
plt.show()
'''
pca = PCA(n_components = 22)
tmp_mat = pca.fit_transform(norm_mat)
#print sum(pca.explained_variance_ratio_)
#pca_mat = pca.fit_transform(norm_mat)

(m,n) = tmp_mat.shape
for i in range(n):
    new_mat = np.delete(tmp_mat,i,1)
    print new_mat.shape
    pca = PCA(n_components=21).fit(new_mat)
    #print i , sum(pca.explained_variance_ratio_)
