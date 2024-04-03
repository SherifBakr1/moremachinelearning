import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

x4_vals = np.load(r'C:\Users\Sheri\Documents\COMP4432\Assignment 5\x4_vals.npy')

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(x4_vals[:, 0], x4_vals[:, 1], s=50)
plt.title('Scatter plot of the data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(x4_vals)
    distortions.append(sum(np.min(cdist(x4_vals, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / x4_vals.shape[0])

plt.subplot(1, 2, 2)
plt.plot(K, distortions, 'bx-')
plt.title('Elbow Method For Optimal k')
plt.xlabel('k')
plt.ylabel('Distortion')

plt.tight_layout()
plt.show()

def cluster_data(method, data, **kwargs):
    if method == 'kmeans':
        model = KMeans(**kwargs).fit(data)
    elif method == 'gmm':
        model = GaussianMixture(**kwargs).fit(data)
    elif method == 'dbscan':
        model = DBSCAN(**kwargs).fit(data)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(**kwargs).fit(data)
    else:
        raise ValueError("Invalid method")
    return model

def plot_clusters(data, model, title):
    if hasattr(model, 'predict'):
        labels = model.predict(data)
    else:
        labels = model.labels_
    
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

kmeans_params = {'n_clusters': 5}
gmm_params = {'n_components': 5}
dbscan_params = {'eps': 0.5, 'min_samples': 5}
agglomerative_params = {'n_clusters': 5}

for method, params in zip(['kmeans', 'gmm', 'dbscan', 'agglomerative'],
                          [kmeans_params, gmm_params, dbscan_params, agglomerative_params]):
    model = cluster_data(method, x4_vals, **params)
    plot_clusters(x4_vals, model, f'{method.capitalize()} Clustering')

