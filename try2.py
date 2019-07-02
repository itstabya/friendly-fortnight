""" 
referred to this link: https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
reinforce with this later please: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

things to better understand
- kmeans++
- random.seed()
"""


from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import math
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('seaborn')

# Importing the dataset
data = pd.read_csv('relative_contributions.csv')
#data.head()

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

from sklearn.cluster import KMeans

k = 5 #number of clusters
kmeans = KMeans(n_clusters=k)

# Fitting the input data
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

centroids = kmeans.cluster_centers_ #centroid values
#plotting the centers
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha = 0.5)
#print("Centroids", centroids) 
plt.show()