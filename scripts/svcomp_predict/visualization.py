
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('ggplot')
# Create color maps        
#cmap = ListedColormap(["#cdc9c9","#e41a1c","#984ea3","#a65628","#377eb8","#ffff33","#4daf4a","#ff7f00"])
cmap = ListedColormap(["#6495ed","#8470ff","#40e0d0","#00ffff","#ffff00","#7cfc00","#ffd700","#d2b48c",
                       "#ffa500","#ff0000","#5569b4"])


class plots(object):
    def __init__(self, x_train, name):
        self.x_tr = x_train
        self.n = name
    
    def scatter_plot(self, data='train'):
        #visualizing the scatter plot for PCA influenced results (dataset and clustering)
        plt.scatter(self.x_tr[:,0], self.x_tr[:,1], marker='.', cmap=cmap)
        plt.title('Visualize '+self.n+' results (' + data + ' data)')
        plt.xlabel(self.n+'1')
        plt.ylabel(self.n+'2')
        plt.show()
        
    def normal_plot(self, param1, param2, string_title):
        plt.plot(param1, param2)
        plt.title(string_title)
        plt.xlabel(self.n+'1')
        plt.ylabel(self.n+'2')
        plt.show()
        
    def sc_plot_with_label(self, y, name, cluster_centers = [],status = 0):
        #visualizing the clusters
        plt.scatter(self.x_tr[y == 0, 0], self.x_tr[y == 0, 1], s = 10, c = 'red', label = 'Arrays')
        plt.scatter(self.x_tr[y == 1, 0], self.x_tr[y == 1, 1], s = 10, c = 'blue', label = 'BitVectors')
        plt.scatter(self.x_tr[y == 2, 0], self.x_tr[y == 2, 1], s = 10, c = 'green', label = 'ControlFlow')
        plt.scatter(self.x_tr[y == 3, 0], self.x_tr[y == 3, 1], s = 10, c = 'cyan', label = 'ECA')
        plt.scatter(self.x_tr[y == 4, 0], self.x_tr[y == 4, 1], s = 10, c = 'magenta', label = 'Floats')
        plt.scatter(self.x_tr[y == 5, 0], self.x_tr[y == 5, 1], s = 10, c = 'pink', label = 'Heap')
        plt.scatter(self.x_tr[y == 6, 0], self.x_tr[y == 6, 1], s = 10, c = 'orange', label = 'Loops')
        plt.scatter(self.x_tr[y == 7, 0], self.x_tr[y == 7, 1], s = 10, c = 'purple', label = 'ProductLines')
        plt.scatter(self.x_tr[y == 8, 0], self.x_tr[y == 8, 1], s = 10, c = 'navy', label = 'Recursive')
        plt.scatter(self.x_tr[y == 9, 0], self.x_tr[y == 9, 1], s = 10, c = 'lime', label = 'Sequentialized')
        plt.scatter(self.x_tr[y == 10, 0], self.x_tr[y == 10, 1], s = 10, c = 'yellow', label = 'DeviceDrivers')
        
        if status == 1:
            plt.scatter(kmeans.cluster_centers_[:,0], cluster_centers[:,1], s = 10,
                        c = 'black', label = 'Centroids')
        #plt.title(name + ' Cluster')
        plt.xlabel(name+'1')
        plt.ylabel(name+ '2')
        plt.legend(loc='best', bbox_to_anchor=(0.5, 1.1),ncol=2, fancybox=True, shadow=True)
        plt.show()

    def sillehoute(self):
        from sklearn.metrics import silhouette_score, silhouette_samples

