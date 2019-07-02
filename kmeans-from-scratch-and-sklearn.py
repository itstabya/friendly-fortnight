from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import math
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('relative_contributions.csv')
data.head()

# Getting the values and plotting it
leftstim = data['LeftStim'].values
rightstim = data['RightStim'].values
X = np.array(list(zip(leftstim, rightstim))) #basically breaking down the data

#Stuff for the scatter plot
plt.scatter(leftstim, rightstim, c='black', s=7)
plt.title("Relative Contributions from Left and Right Stimulus")
"""behavioural variables that we currently have
#factors = ['LeftStim','RightStim','Reward','Choice','PrevChoice', 'Difficulty','Movement']"""
plt.xlabel("Left Stimulus")
plt.ylabel("Right Stimulus")

# Euclidean Distance Calculator
def dist(a, b, ax=1):
    print()
    print('dis be a-b', a-b)
    print()
    ans = np.linalg.norm(a - b, axis=ax)
    if ans is None:
        print('ans be broken HELP')
    else:
        return ans

# Number of clusters
k = 4
C_x = np.random.random_sample(k)
C_y = np.random.random_sample(k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32 )

#Testing with some centroids I know always fail
#C = np.array([[0.10260828,0.3556745],[0.05638211, 0.05612206],
#[0.39425027, 0.13211249]], dtype=np.float32)
print("Initial Centroids", C)

# Plotting along with the Centroids
plt.scatter(leftstim, rightstim, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)

# Cluster Labels(0, 1, 2)
clusters = np.zeros(len(X))

# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)

# Loop will run till the error becomes zero
cluster_count = 0

"""
General idea of k-means clustering algorithm:
1. Randomly pick K cluster centers - C is the set of all the coords we have for our centroids
2. Assign each input value to the closest centroid (using our dist function)
3. Find new centroid by taking average of all the points in that cluster
4. Repeat steps 2 and 3 until none of the cluster assignments change. 
    aka! our clusters remain stable

How do we choose k?
- elbow method
- avg silhouette
- gap statistic ("more sophisticated")

these are centroids that don't work
[[0.10260828 0.3556745 ]
 [0.05638211 0.05612206]
 [0.39425027 0.13211249]]

"""
while error != 0:

    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        #print("distances" , distances)
        cluster = np.argmin(distances) #returns index of minimum distance
        #print('cluster assignment', cluster)
        clusters[i] = cluster
        #print("cluster assignment", clusters[i]) #cluster assignment
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        for point in points:
            print(point)
        print(len(C))
        C[i] = np.mean(points, axis=0)

    error = dist(C, C_old, None)
    print('every iteration of error', error)
    cluster_count += 1

print("Counting how many times we reassign", cluster_count)
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

'''
==========================================================
scikit-learn
==========================================================
'''

from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

# Comparing with scikit-learn centroids
print("Centroid values")
print("Scratch")
print(C) # From Scratch
print("sklearn")
print(centroids) # From sci-kit learn
plt.show()