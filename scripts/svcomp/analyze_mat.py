from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics

class ana():
    def __init__(self,mat,clusterL):
        self.ipmat = mat
        self.cl = clusterL

    def mat_clus(self):
        gl = self.ipmat[:,-1]
        #print type(gl), '\n', gl
        gl = np.squeeze(np.asarray(gl))
        #print type(gl), '\n', gl
        print gl
        gl_mat = self.generateMat(gl,3)


    def generateMat(self,vec,n):
        A = np.empty([n+1,np.size(vec)])
        for i in range(n+1):
            tmp = vec
            tmp[(tmp > i) | (tmp < i)] = n+1
            np.putmask(tmp,tmp==i,0)
            A[i,:] = tmp
        #A[A == (n+1)] = 1
        print A
        return A
