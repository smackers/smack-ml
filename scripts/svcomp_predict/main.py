
# coding: utf-8

# In[1]:


from ml_models import supervised, unsupervised, dim_red
from data_cleaning import preprocessing
from visualization import plots
import _pickle as pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, homogeneity_completeness_v_measure
import matplotlib.pyplot as plt
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')

dataset = preprocessing().load_data()
dataset = preprocessing().missing_data(dataset)
dataset.to_csv('dataset.csv')

X = dataset.iloc[:,1:-1].values #features matrix
y = dataset.iloc[:,-1].values #labels vector

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 10)

#normalize the dataset
X_train, X_test = preprocessing().feature_scaling(X_train, X_test)


#Achieve max retention/ variance ratio around dim = 21. So we can easily perform
#dim reduction to 21 without loosing any information or performance of the ML
#algorithm in later stages
obj = dim_red(X_train, X_test)

retention = []
dimensions = [i for i in range(1, len(dataset.columns)-1)]
for i in range(1,len(dataset.columns)-1):
    X_train, X_test, variance_ratio = obj.pca_compute(i)
    retention.append(sum(variance_ratio))
    
plt.plot(dimensions, retention)
#plt.title(string_title)
plt.xlabel('# of dimensions')
plt.ylabel('variance ratio')
plt.show()
# In[6]:


mode = int(input("Choose an algorithm \n"
                "1. PCA \n 2. Supervised Learning \n 3. Unsupervised Learning \n"))
alg = ''

if mode == 1:
    obj_model = dim_red(X_train, X_test)
    alg = 'PCA'
    #As concluded above, 21 dims are enough to retain most of the information
    #captured by the feature vectors of size 33
    X_train, X_test, variance_ratio = obj_model.pca_compute(21)
    
    print(sum(variance_ratio))
    #dataset.to_csv('dataset_pca.csv')
#Add the code for visualizing PCA in 2D for 21 dims input and then
#apply t-SNE on PCA (21 dims) and visualize the results. t-SNE is able to capture
#non-linear relationships between features while performing dim reduction so 
#expected results should be better than PCA (although t-SNE is very slow)

# In[8]:


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


# In[9]:


if mode == 3:
    X_train, X_test = preprocessing().feature_scaling(X)
    
    if mode_2 == 1 and num_of_dim == 2:
        if mode_2 == 1:
            X_train, X_test, variance_ratio = dim_red(X_train).pca_compute(num_of_dim)
        else:
            X_train, X_test, variance_ratio = dim_red(X_train).lda_compute(num_of_dim, y)
    
    print(variance_ratio)
    
    obj_model = unsupervised(X_train, y)
    
    choice = int(input("Which model do you want to run \n 1. K-Means \n 2. Hierarchical \n 3. Spectral \n"))
    status = 0; kcenter = [];
    if choice == 1:
        sub_choice = int(input("Which strategy do you want to use \n 1. Selective cluster centers \n 2. k-means++ \n"))
        if sub_choice == 1:
            no_of_clusters = [i for i in range(1,12)];
            center_filtered = np.vstack((X[y == 0][:1], X[y == 1][:1], X[y == 2][:1], X[y == 3][:1],
                                    X[y == 4][:1], X[y == 5][:1], X[y == 6][:1], X[y == 7][:1],
                                   X[y == 8][:1], X[y == 9][:1], X[y == 10][:1], X[y == 11][:1]))
            strategy = np.array(center_filtered)
            #print(strategy[0:2].reshape(2,-2))

        else: strategy = 'k-means++';
        algo = 'KMeans'; status = 1;
        y_pred, wcss, kcenter = obj_model.kmean(strategy)
        obj_viz = plots(X_train)
        obj_viz.normal_plot(no_of_clusters,wcss, 'WCSS for '+algo, 'Number of Clusters', 'WCSS')
            
    elif choice == 2:
        algo = 'Agglomerative';
        y_pred = obj_model.agglomerative()
        
    else:
        algo = 'Spectral'; status = 0;
        y_pred = obj_model.spectral()


# In[ ]:


print(alg, algo)
print(len(alg))


# In[ ]:


if len(alg)>0:
    plots(X_train,alg).scatter_plot('entire')


# In[ ]:


plots(X_train,alg).sc_plot_with_label(y, algo, kcenter, status)
plots(X_train,alg).sc_plot_with_label(y_pred, algo, kcenter, status)

