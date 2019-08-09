
# coding: utf-8

# In this Notebook, we compare the effects of Standardization vs Normalization on the dataset. Our goal is to apply PCA to the dataset and it is a common practice to apply standardization before PCA (not normalization).

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from data_cleaning import preprocessing
from sklearn.decomposition import PCA
from ml_models import dim_red
import seaborn as sns

def plot_curves(new_dataset):
    retention = []
    dimensions = [i for i in range(1, 32)]
    for i in range(1,32):
        X_train, X_test, variance_ratio = obj.pca_compute(i)
        retention.append(sum(variance_ratio))
    
    plt.plot(dimensions, retention)
    plt.title('PCA curve')
    plt.xlabel('# of dimensions')
    plt.ylabel('variance ratio')
    plt.show()
    
def heatmaps(df):
    df_ = pd.DataFrame(df, columns=None)
    corr_ = df_.corr()
    ax = sns.heatmap(corr_, vmin=-1, vmax=1, center=0,
                     cmap=sns.diverging_palette(20, 220, n=200),square=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')

# In[2]:
#Dealing with missing data (Replace with the mean)

dataset = preprocessing().load_data()
dataset = preprocessing().missing_data(dataset)
#dataset.to_csv('dataset.csv')
new_dataset = dataset.iloc[:,1:-1]
labels = dataset.iloc[:,-1] #labels vector
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
heatmaps(new_dataset)

# In[3]:

cols = new_dataset.columns
print(len(cols))


# In[4]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# In[5]:
#standardization

from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


# In[6]:

'''
minmax_scale = preprocessing.MinMaxScaler().fit(X_train)
X_train_minmax = minmax_scale.transform(X_train)
X_test_minmax = minmax_scale.transform(X_test)
'''

# In[7]:

#apply PCA to non-standardized data
pca = PCA(n_components=2).fit(X_train)
print(sum(pca.explained_variance_ratio_))
X_train_pc = pca.transform(X_train)
X_test_pc = pca.transform(X_test)


# In[8]:

#apply PCA to standardized data
pca_std = PCA(n_components=2).fit(X_train_std)
print(sum(pca_std.explained_variance_ratio_))
X_train_std_pca = pca_std.transform(X_train_std)
X_test_std_pca = pca_std.transform(X_test_std)

obj = dim_red(X_train_std, X_test_std)
plot_curves(X_train_std)

# In[9]:


print(labels.nunique())
print(labels.value_counts())
# In[11]:


co = ("#6495ed","#8470ff","#40e0d0","#00ffff","#ff1493","#7cfc00","#ffd700","#d2b48c",
        "#ffa500","#ff0000","#5569b4")
        
#black, red, deep pink, gold, maroon, green, blue, fuchsia, purple, navy, gray ,salmon
#co = ("#000000" ,"#ff0000", "#ff1493", "#ffd700", "#800000", "#008000", "#0000ff", "#ff00ff", "#800080","#000080","#808080", "#ffa07a")
fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1, figsize=(10,4))
for l,c in zip(range(1,12),co):
    ax1.scatter(X_train_pc[y_train==l,0], X_train_pc[y_train==l,1],
               color=c,
               label='class %s'%l, alpha=0.5,marker='.')
for l,c in zip(range(1,12),co):
    ax2.scatter(X_train_std_pca[y_train==l,0], X_train_std_pca[y_train==l,1],
               color=c,
               label='class %s'%l, alpha=0.5,marker='.')
    
ax1.set_title('Original dataset after PCA')
ax2.set_title('Dataset after standardization and PCA')

for ax in (ax1,ax2):
    ax.set_xlabel('1st component')
    ax.set_ylabel('2nd component')
    #ax.legend(loc='upper right')
    #ax.grid()

plt.legend(loc='best', bbox_to_anchor= (0.8, 0.5, 0.5, 0.3), ncol=1, borderaxespad=0, frameon=False)
#handles, labels = ax.get_legend_handles_labels()
#plt.legend(handles, labels,loc='right')
#ax3.legend((loc='upper right')
plt.tight_layout()
plt.show()


# In[12]:
'''

co = ("#6495ed","#8470ff","#40e0d0","#00ffff","#ffff00","#7cfc00","#ffd700","#d2b48c",
        "#ffa500","#ff0000","#5569b4")
fig, (ax1,ax2,ax3) = plt.subplots(ncols=3,nrows=1, figsize=(10,4))
for l,c in zip(range(1,12),co):
    ax1.scatter(X_test_pc[y_test==l,0], X_test_pc[y_test==l,1],
               color=c,
               label='class %s'%l, alpha=0.5,marker='.')
for l,c in zip(range(1,12),co):
    ax2.scatter(X_test_std_pca[y_test==l,0], X_test_std_pca[y_test==l,1],
               color=c,
               label='class %s'%l, alpha=0.5,marker='.')
for l,c in zip(range(1,12),co):
    ax3.scatter(X_test_minmax_pca[y_test==l,0], X_test_minmax_pca[y_test==l,1],
               color=c,
               label='class %s'%l, alpha=0.5,marker='.')
    
ax1.set_title('Non-scaled test dataset after PCA')
ax2.set_title('Standardized test dataset after PCA')
ax3.set_title('Normalized test dataset after PCA')

for ax in (ax1,ax2,ax3):
    ax.set_xlabel('1st component')
    ax.set_ylabel('2nd component')
    ax.legend(loc='upper right')
    ax.grid()
plt.tight_layout()
plt.show()
'''

# In the next few steps, we want to compare the results of PCA, t-SNE and PCA with t-SNE. We will use the standardized training and test data from above.

# In[13]:


from sklearn.manifold import TSNE
#directly apply TSNE
df_tsne=TSNE(n_iter=300, n_components=2)
X_train_tsne = df_tsne.fit_transform(X_train_std)
X_test_tsne = df_tsne.fit_transform(X_test_std)

#apply TSNE after PCA - 21 dims
pca = PCA(n_components=21).fit(X_train_std,y_train)
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
tsne = TSNE(n_iter=300, n_components=2)
X_train_ptsne = tsne.fit_transform(X_train_pca)
X_test_ptsne = tsne.fit_transform(X_test_pca)


# In[14]:


co = ("#6495ed","#8470ff","#40e0d0","#00ffff","#ffff00","#7cfc00","#ffd700","#d2b48c",
        "#ffa500","#ff0000","#5569b4")
fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1, figsize=(10,4))
for l,c in zip(range(1,12),co):
    ax1.scatter(X_train_tsne[y_train==l,0], X_train_tsne[y_train==l,1],
               color=c,
               label='class %s'%l, alpha=0.5,marker='.')
for l,c in zip(range(1,12),co):
    ax2.scatter(X_train_ptsne[y_train==l,0], X_train_ptsne[y_train==l,1],
               color=c,
               label='class %s'%l, alpha=0.5,marker='.')
    
ax1.set_title('t-SNE on original dataset')
ax2.set_title('t-SNE dataset after optimal PCA')

for ax in (ax1,ax2):
    ax.set_xlabel('1st component')
    ax.set_ylabel('2nd component')
    #ax.legend(loc='upper right')
    ax.grid()

plt.legend(loc='best', bbox_to_anchor= (0.8, 0.5, 0.5, 0.3), ncol=1, borderaxespad=0, frameon=False)
plt.tight_layout()
plt.show()