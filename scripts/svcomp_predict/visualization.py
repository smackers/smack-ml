
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
#plt.figure(figsize=(20,16))
plt.rcParams['figure.figsize'] = (15, 12)
plt.style.use('ggplot')

# Create color maps        
cmap = ["#6495ed","#8470ff","#40e0d0","#00ffff","#ffff00","#7cfc00","#ffd700","#d2b48c",
                       "#ffa500","#ff0000","#5569b4"]
#sns.set_palette(cmap)

class plots(object):
    def __init__(self, x_train, name):
        self.x_tr = x_train
        self.n = name
    
    def scatter_plot(self, data='train'):
        #visualizing the scatter plot for PCA influenced results (dataset and clustering)
        plt.scatter(self.x_tr[:,0], self.x_tr[:,1], marker='.', cmap=cmap)
        plt.title('Visualize '+self.n+' results (' + data + ' data)')
        
        min_x, max_x = int(min(self.x_tr[:,0])), int(max(self.x_tr[:,0]))
        min_y, max_y = int(min(self.x_tr[:,1])), int(max(self.x_tr[:,1]))
        plt.xlim([min_x,max_x])
        plt.ylim([min_y, max_y])
        plt.xticks(np.arange(min_x,max_x,1.5))
        plt.yticks(np.arange(min_y,max_y,1.5))
        
        plt.xlabel(self.n+'1')
        plt.ylabel(self.n+'2')
        plt.show()
        
    def normal_plot(self, param1, param2, string_title, x_ax, y_ax):
        plt.plot(param1, param2)
        plt.title(string_title)
        plt.xlabel(x_ax)
        plt.ylabel(y_ax)
        plt.show()

    def sc_plot_with_label(self, y, name, cluster_centers = [],status = 0):
        sns.set()
        fscet = sns.lmplot(data=self.x_tr, x = 'x1', y = 'x2',
                           hue=y,fit_reg = False, legend_out=True, row=y,
                           palette = cmap)
        
        min_x, max_x = int(self.x_tr['x1'].min()), int(self.x_tr['x1'].max())
        min_y, max_y = int(self.x_tr['x2'].min()), int(self.x_tr['x2'].max())
        plt.xlim(min_x,max_x)
        plt.ylim(min_y,max_y)
        plt.xticks(np.arange(min_x,max_x,7.5))
        plt.yticks(np.arange(min_y,max_y,12.5))
        
        '''leg = fscet.ax.legend(loc='best',bbox_to_anchor=(0.5, 1.1),ncol=4, fancybox=True, title='label')
        for i, text in enumerate(leg.get_texts()):
            plt.setp(text,color=cmap[i])'''
        plt.show()
        
    def sillehoute(self):
        from sklearn.metrics import silhouette_score, silhouette_samples
        range_n_clusters = [2, 3, 4, 5, 6]


        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values =                     sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette coefficient of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

            plt.show()

