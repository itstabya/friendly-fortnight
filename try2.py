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



'''
==========================================================
scikit-learn
==========================================================
'''

from sklearn.cluster import KMeans

# Number of clusters
k = 5
kmeans = KMeans(n_clusters=k)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Centroid values
centroids = kmeans.cluster_centers_
#plotting the centers
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha = 0.5)

# Comparing with scikit-learn centroids
print("Centroids", centroids) # From sci-kit learn
plt.show()