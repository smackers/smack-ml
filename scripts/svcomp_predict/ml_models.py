
# coding: utf-8

# In[1]:


#!/usr/bin/env python3


# In[2]:


from sklearn.metrics import accuracy_score, confusion_matrix, homogeneity_completeness_v_measure
class supervised(object):
    def __init__(self, X_train, y_train, X_test=None):
        self.x_tr = X_train
        self.x_te = X_test
        self.y_tr = y_train
        
    def knn(self, neighbor):
        from sklearn.neighbors import KNeighborsClassifier
        predictions = []; n = [];
        for i in range(neighbor+1):
            knn = KNeighborsClassifier(n_neighbors=i+1)
            knn.fit(self.x_tr, self.y_tr)
            y_pred = knn.predict(self.x_te)
            n.append(i+1)
            predictions.append(y_pred)
            
        return predictions,n
    
    def log_reg(self):
        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        n = [0]; predictions = [];
        classifier = LogisticRegression()
        classifier.fit(self.x_tr, self.y_tr)
        y_pred = classifier.predict(self.x_te)
        predictions.append(y_pred)
        
        return predictions,n
    
    def rand_forest(self):
        # Fitting Random Forest Classification to the Training set
        from sklearn.ensemble import RandomForestClassifier
        predictions = []; config = [];
        for i in range(10,101,10):
            for j in range(3,31,3):
                classifier = RandomForestClassifier(n_estimators = i, max_depth = j, criterion = "entropy")
                classifier.fit(self.x_tr, self.y_tr)
                y_pred = classifier.predict(self.x_te)
                predictions.append(y_pred)
                config.append((i,j))
        
        return predictions, config
    
    def svm(self):
        from sklearn.svm import SVC
        predictions = []; c = [0.001, 0.01, 0.1, 1, 10, 100];
        for i in c:
            clf = SVC(C = i, kernel = 'rbf')
            clf.fit(self.x_tr, self.y_tr)
            y_pred = clf.predict(self.x_te)
            predictions.append(y_pred)
            
        return predictions, c
    
    


# In[3]:


class dim_red(object):
    def __init__(self, x_train, x_test=[]):
        self.x_tr = x_train
        self.x_te = x_test
        
    def pca_compute(self,i):
        # Applying PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components = i)
        x_train = pca.fit_transform(self.x_tr)
        if len(self.x_te) > 0: x_test = pca.transform(self.x_te);
        else: x_test = self.x_te;
        var_explained = pca.explained_variance_ratio_
        
        return x_train, x_test, var_explained
    
    def lda_compute(self, i, y_true):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = i)
        x_train = lda.fit_transform(self.x_tr, y_true)
        if len(self.x_te) > 0: x_test = lda.transform(self.x_te);
        else: x_test = self.x_te;
        var_explained = lda.explained_variance_ratio_
            
        return x_train, x_test, var_explained


# In[4]:


class unsupervised(object):
    def __init__(self, features, labels):
        self.f = features
        self.l = labels
        
    def kmean(self, strategy, i=12):
        wcss = []
        #Applying K-Means clustering
        from sklearn.cluster import KMeans
        for i in range(2,i):
            kmeans = KMeans(n_clusters=i, init=strategy[i], max_iter=100, n_init = 10,random_state=0).fit(self.f)
            y_pred = kmeans.labels_
            wcss.append(kmeans.inertia_)
            c_center = kmeans.cluster_centers_
        
        print(homogeneity_completeness_v_measure(self.l,y_pred))
        return y_pred, wcss, c_center
    
    def agglomerative(self, i=12):
        from sklearn.cluster import AgglomerativeClustering
        for i in range(2,i):
            agg = AgglomerativeClustering(n_clusters=i, linkage = 'single').fit(self.f)
            y_pred = agg.labels_
        
        print(homogeneity_completeness_v_measure(self.l,y_pred))
        return y_pred
        
    def spectral(self, i = 12):
        from sklearn.cluster import SpectralClustering
        for i in range(2,i):
            spc = SpectralClustering(n_clusters=i, affinity='nearest_neighbors', n_neighbors=8,
                                    assign_labels='discretize', random_state=0).fit(self.f)
            y_pred = spc.labels_
            c_center = spc.

        print(homogeneity_completeness_v_measure(self.l,y_pred))
        return y_pred

