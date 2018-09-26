import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from copy import deepcopy

#function to plot with centroids
def plot_centroids(f1,f2,centroids,labels):
	colors = ['r', 'g', 'b']
	fig, ax = plt.subplots()
	for i in range(3):
		points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
		ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
	ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
	plt.title("Current positions of centroids")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.show()

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

data = pd.read_csv('kData.csv')

f1 = data['X'].values
f2 = data['Y'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)
plt.title("Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

k = 3
#Initializing centroids
C_x = np.random.randint(0, np.max(X), size=k)
C_y = np.random.randint(0, np.max(X), size=k)
centroids = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial centroids::",centroids)

old_centroids = np.zeros(centroids.shape)
labels = np.zeros(len(X))
error = dist(centroids, old_centroids)
while error.all() != 0:
	#Assigning cluster
	for i in range(len(X)):
		distances = dist(X[i], centroids)
		label = np.argmin(distances)
		labels[i] = label
	plot_centroids(f1,f2,centroids,labels)
	#Storing the old centroids
	old_centroids = deepcopy(centroids)
	#Resetting centroids
	for i in range(k):
		points = [X[j] for j in range(len(X)) if labels[j] == i]
		centroids[i] = np.mean(points, axis=0)
	error = dist(centroids, old_centroids)

print("Final Centroids::",centroids)