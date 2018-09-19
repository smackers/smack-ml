from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics

class analysis():
    def __init__(self, kmeans, norm_mat):
        self.kmeans = kmeans
        self.norm_mat = norm_mat

    def clustering(self,cl):
        #print kmeans.labels_, '\n' , norm_mat[:,-1]
        #print map(int,np.ravel(norm_mat[:,-1]).tolist())

        gl = map(int,np.ravel(self.norm_mat[:,-1]).tolist())

        print metrics.adjusted_mutual_info_score(gl,cl)
        print metrics.normalized_mutual_info_score(gl,cl)
        print metrics.fowlkes_mallows_score(gl,cl)

        cd = {}; gd = {}
        for i in range(len(cl)):
        	if cl[i] not in cd:
        		cd[cl[i]] = [i]
        	else: cd[cl[i]].append(i)
        	if gl[i] not in gd:
        		gd[gl[i]] = [i]
        	else: gd[gl[i]].append(i)

        print cd.keys(), gd.keys()
        print gd #cd,'\n',gd
        sum = 0
        for item1 in cd:
        	tmp = [];
        	for item2 in gd:
        		count = 0
        		for i in range(len(cd[item1])):
        			for j in range(len(gd[item2])):
        				if cd[item1][i] == gd[item2][j]:
        					count += 1
        		tmp.append(count)

        	t = max(tmp)

        	sum = sum + t

        #print len(cl), sum
        #print (sum*100)/len(cl)
