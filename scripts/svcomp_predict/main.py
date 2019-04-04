
# coding: utf-8

# In[ ]:


from ml_models import supervised, unsupervised, dim_red
from data_cleaning import preprocessing
from visualization import plots
from sklearn.metrics import accuracy_score, confusion_matrix, homogeneity_completeness_v_measure
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset = preprocessing().load_data()
dataset = preprocessing().missing_data(dataset)


# In[ ]:


X = dataset.iloc[:,1:-1].values #features matrix
y = dataset.iloc[:,-1].values #labels vector


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 10)


# In[ ]:


#normalize the dataset
X_train, X_test = preprocessing().feature_scaling(X_train, X_test)


# In[ ]:


mode = int(input("1. PCA \n 2. LDA \n 3. Supervised Learning \n 4. Unsupervised Learning \n"))
mode_2 = mode


# In[ ]:


if mode == 1 or mode == 2:
    obj_model = dim_red(X_train, X_test)
    num_of_dim = int(input("Enter number of dimensions to feed="))
    if mode == 1:
        alg = 'PCA'
        X_train, X_test, variance_ratio = obj_model.pca_compute(num_of_dim)
    elif mode == 2:
        alg = 'LDA'
        X_train, X_test, variance_ratio = obj_model.lda_compute(num_of_dim, y_train)
    
    if num_of_dim == 2:
        obj_viz = plots(X_test,alg+'1', alg+'2')
        obj_viz.scatter_plot()
        
    mode = int(input("2. Supervised Learning \n 3. Unsupervised Learning \n"))


# In[ ]:


if mode == 2:
    obj_model = supervised(X_train, y_train, X_test)
    choice = int(input("which model do you want to run \n 1.KNN \n 2. Logistic regression \n 3. Random Forest \n 4. SVM \n 5. ANN \n"))
    if choice == 1:
        algo = 'knn'
        y_pred_list, param = obj_model.knn(6)
    elif choice == 2:
        algo = 'logit'
        y_pred_list, param = obj_model.log_reg()
    elif choice == 3:
        algo = 'random forest';
        y_pred_list, param = obj_model.rand_forest()
        param_pair = param
        param = [x[0]+x[1] for x in param_pair]
            
    elif choice == 4:
        algo = 'SVM'
        y_pred_list, param = obj_model.svm()
        
    accuracy = []
    
    for i in range(len(y_pred_list)):
        accuracy.append(accuracy_score(y_test,y_pred_list[i])*100)

    i = accuracy.index(max(accuracy))
    best_vals = param[i]
    print("The algorithm is most optimized for {0}".format(best_vals))
        
    obj_viz = plots(X_train)
    obj_viz.normal_plot(param,accuracy, 'Accuracy for '+algo, 'parameters', 'accuracy')


# In[ ]:


if mode == 3:
    X, X_te = preprocessing().feature_scaling(X)
    
    if mode_2 == 1 and num_of_dim == 2:
        if mode_2 == 1:
            X, X_test, variance_ratio = dim_red(X).pca_compute(num_of_dim)
        else:
            X, X_test, variance_ratio = dim_red(X).lda_compute(num_of_dim, y)
            
    #plots(X,alg+'1',alg+'2').scatter_plot('whole (original)', y, 'nn')
    
    print(variance_ratio)
    
    obj_model = unsupervised(X, y)
    
    choice = int(input("Which model do you want to run \n 1. K-Means \n 2. Hierarchical \n 3. Spectral \n"))
    no_of_clusters = [i for i in range(1,12)];
    if choice == 1:
        sub_choice = int(input("Which strategy do you want to use \n 1. Selective cluster centers \n 2. k-means++ \n"))
        if sub_choice == 1:
            center_filtered = np.vstack((X[y == 0][:1], X[y == 1][:1], X[y == 2][:1], X[y == 3][:1],
                                    X[y == 4][:1], X[y == 5][:1], X[y == 6][:1], X[y == 7][:1],
                                   X[y == 8][:1], X[y == 9][:1], X[y == 10][:1], X[y == 11][:1]))
            strategy = np.array(center_filtered)
            #print(strategy[0:2].reshape(2,-2))
            #print(shape(strategy[0]))
        else: strategy = 'k-means++';
        algo = 'KMeans'
        y_pred, wcss = obj_model.kmean(strategy)
        obj_viz = plots(X_train)
        obj_viz.normal_plot(no_of_clusters,wcss, 'WCSS for '+algo, 'Number of Clusters', 'WCSS')
            
    elif choice == 2:
        algo = 'Agglomerative'
        y_pred = obj_model.agglomerative()
        
    else:
        algo = 'Spectral'
        y_pred = obj_model.spectral()

    plots(X,alg+'1',alg+'2').scatter_plot('whole (original)', y, algo)
    plots(X,alg+'1',alg+'2').scatter_plot('whole (predicted)', y_pred, algo)

