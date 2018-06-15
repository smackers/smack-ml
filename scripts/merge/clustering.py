import numpy as np
import pickle, glob2
from sklearn.cluster import KMeans

'''Goal: Perform clustering
1. Passing the centers
    a) Plot the #iteration vs accuracy curve
    b) store the centers for best clustering results
'''
class need_for_clustering():
    def __init__(self, mat, end):
        self.mat = mat
        (m,n) = self.mat.shape
        self.end = end

    def select_features(self):
        tmp = [[] for j in range(self.end)];
        for i in range(m):
            start = 0;
            while self.mat[i,-1] != start:
                start += 1
            tmp[start].append(self.mat[i,:])
        print tmp[0].shape
        print tmp[1][0]
        #if len(tmp) == self.end: print "Yahoo"
        return tmp

    def compute_centers(self, p):
    #return a kxn matrix for k-cluster centers
        tmp = self.select_features()
        final = []
        for i in range(len(tmp)):
            np.random.shuffle(tmp[i])
            compute_mat = tmp[i][:p,:]
            (a,b) = compute_mat.shape; vec = [];
            for j in range(b-1):
                vec.append(np.mean(compute_mat[:,i]))
            final.append(vec)
        print final.shape
        return final


if __name__ == '__main__':
    input_mat = pickle.load(open('normalizedMergedLabels.txt','r'))
    (m,n) = input_mat.shape

    select_centers = need_for_clustering(input_mat, 11)
    centers = select_centers.compute_centers(3)

    #kmeans_output = KMeans(n_clusters=11,random_State=0).fit(cluster_mat)
    #cl = kmeans.labels_
