import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# Load the dataset
x1_vals = np.load(r'C:\Users\Sheri\Documents\COMP4432\Assignment 5\x1_vals.npy')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x1_vals[:, 0], x1_vals[:, 1])
plt.title("Data Scatter Plot")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

distortions = []
K_range = range(1, 11)
for k in K_range:
    kmeanModel = KMeans(n_clusters=k, random_state=42)
    kmeanModel.fit(x1_vals)
    distortions.append(kmeanModel.inertia_)

plt.subplot(1, 2, 2)
plt.plot(K_range, distortions, 'bx-')
plt.title('Elbow Method For Optimal k')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.show()

silhouette_scores = []
for k in K_range:
    if k == 1: continue  
    kmeanModel = KMeans(n_clusters=k, random_state=42)
    kmeanModel.fit(x1_vals)
    silhouette_scores.append(silhouette_score(x1_vals, kmeanModel.labels_))

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, 'bx-')
plt.title('Silhouette Method For Optimal k')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.show()

chosen_k = 4  
kmeans = KMeans(n_clusters=chosen_k, init='random', n_init='auto', max_iter=300, random_state=42)
kmeans.fit(x1_vals)

plt.figure(figsize=(10, 5))
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for i in range(chosen_k):
    plt.scatter(x1_vals[kmeans.labels_ == i, 0], x1_vals[kmeans.labels_ == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='gray', label='Centroids', marker='*')
plt.title("KMeans Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

print(f"A. Methods for selecting K: Elbow method and Silhouette score. These methods are popular for their simplicity and effectiveness in identifying a good balance of cluster compactness and separation.")
print(f"B. Selected value for K: {chosen_k}. This choice is supported by visual inspection of the Elbow and Silhouette plots, suggesting a balance of within-cluster variance reduction and cluster separation.")
print(f"C. Iterations for convergence: {kmeans.n_iter_}.")
print(f"D. Cluster centroids: {kmeans.cluster_centers_}.")
print(f"E. Measure for cluster coherence: Inertia, or within-cluster sum of squares. Value: {kmeans.inertia_}. This value represents the total variance within the clusters; lower values indicate tighter, more coherent clusters, suggesting a good clustering fit to the data.")

#part b

range_n_clusters = [2, 3, 4, 5, 6]

plt.figure(figsize=(len(range_n_clusters) * 6, 8))
for i, n_clusters in enumerate(range_n_clusters):
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(x1_vals)

    silhouette_avg = silhouette_score(x1_vals, cluster_labels)
    sample_silhouette_values = silhouette_samples(x1_vals, cluster_labels)

    plt.subplot(1, len(range_n_clusters), i + 1)
    y_lower = 10
    for j in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
        ith_cluster_silhouette_values.sort()
        size_cluster_j = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_j

        color = cm.nipy_spectral(float(j) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
        y_lower = y_upper + 10  

    plt.title("Silhouette plot for {} clusters".format(n_clusters))
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])  # Clear the y-axis labels / ticks
    plt.xticks(np.arange(-0.1, 1.1, 0.2))
    plt.xlim([-0.1, 1.0])

plt.tight_layout()
plt.show()

X, y_true = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=42, n_features=2)

kmeans = KMeans(n_clusters=5, random_state=42)
y_pred = kmeans.fit_predict(X)

ari_score = adjusted_rand_score(y_true, y_pred)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis')
plt.title('Actual Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', alpha=0.5, label='Centroids')
plt.title('Predicted Clusters by KMeans')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

