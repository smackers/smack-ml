
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
cmap = ListedColormap(["#ffff00","#6495ed","#8470ff","#40e0d0","#00ffff","#7cfc00","#ffd700","#d2b48c",
                       "#ffa500","#ff0000","#5569b4"])


class plots(object):
    def __init__(self, x_train, x_string, y_string):
        self.x_tr = x_train
        self.x_str = x_string
        self.y_str = y_string
    
    def scatter_plot(self, data='train', y_pred=None, alg = 'PCA'):
        #visualizing the scatter plot for PCA influenced results (dataset and clustering)
        plt.scatter(self.x_tr[:,0], self.x_tr[:,1], c=y_pred, marker='.', cmap=cmap)
        plt.title('Visualize '+alg+' results (' + data + ' data)')
        plt.xlabel(self.x_str)
        plt.ylabel(self.y_str)
        plt.show()
        
    def normal_plot(self, param1, param2, string_title):
        plt.plot(param1, param2)
        plt.title(string_title)
        plt.xlabel(self.x_str)
        plt.ylabel(self.y_str)
        plt.show()

    def sillehoute(self):
        from sklearn.metrics import silhouette_score, silhouette_samples

