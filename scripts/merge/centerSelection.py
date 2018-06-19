from sklearn.cluster import KMeans
import numpy as np
import pickle

path = 'outputCategory/'



if __name__ == '__main__':
    inp_mat = pickle.load(open('normalizedMergedLabels.txt','r'))
    (m,n) = inp_mat.shape
    cluster_mat = inp_mat[:,0:n-1]
    original_labels = (inp_mat[:,-1].astype(int)).tolist()
    print original_labels[1][0]
    print len(original_labels)

    clusterm_mat = cluster_mat.tolist()
    inp_mat = inp_mat.tolist()


    '''generate cluster centers'''
    tmp = [[] for j in range(11)]
    #print inp_mat[0],'\n', len(inp_mat)
    for i in range(len(inp_mat)):
        num = int(inp_mat[i][-1])
        tmp[num].append(inp_mat[i])

    clus_center = []
    for i in range(len(tmp)):
        tmp[i] = np.matrix(tmp[i])
        np.random.shuffle(tmp[i])
        t = tmp[i][0:3,:]
        m = t.mean(0).tolist()
        clus_center.append(m[0])

    '''run clustering with computed seed'''
    clus_center = np.asarray(clus_center)
    (a,b) = clus_center.shape
    clus_center = clus_center[:,0:b-1]

    accuracy = []; iter = [];
    for i in range(800):
        count = 0;
        iter.append(i)

        kmeans = KMeans(n_clusters=11,init=clus_center,n_init=1,max_iter=1).fit(cluster_mat)
        cl = kmeans.labels_
        clus_center = kmeans.cluster_centers_
        #print cl[0]

        if len(original_labels) == len(cl):
            for i in range(len(original_labels)):
                if original_labels[i][0] == cl[i]:
                    count += 1
        accuracy.append((count*100)/len(original_labels))

    
    #'''
