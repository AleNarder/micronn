import numpy as np
from sklearn.datasets import make_blobs
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate points from 3 Gaussian blobs
centers = [(0, 0, 0), (2, 2, 2), (-2, -2, -2)]  # Centers of the Gaussians
n_samples = 1000  # Total number of samples
cluster_std = [0.75, 0.95, 0.85]  # Standard deviation for each Gaussian

# Use make_blobs to generate the 3D points
X, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, n_features=3, random_state=42)

# Plotting the 3D scatter of the points
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with colors based on the Gaussian each point comes from
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=20)

ax.set_title('3D Points from Three Gaussian Distributions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

dataset = []
for x, label in zip(X, labels):
    dataset.append({
        "data": x.tolist(), 
        "label" : int(label)
})

json.dump(dataset, open("clusters.json", "w"))
    