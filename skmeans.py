import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv('kData.csv')
f1 = data['X'].values
f2 = data['Y'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)
plt.title("Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
print("Centroids::\n",centroids)
colors = ['r', 'g', 'b']
fig, ax = plt.subplots()
for i in range(3):
	points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
	ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.title("Clusters found by K-means")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()