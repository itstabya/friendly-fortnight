""" 
referred to this link: https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
reinforce with this later please: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
in considering other clustering approaches: https://www.datascience.com/blog/k-means-alternatives

things to better understand
- kmeans++
- random.seed()
- SpectralClusering

considering gaussian mixture models (better quantitative measure of fitness per k)
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('seaborn')

# Importing the dataset
data = pd.read_csv('relative_contributions.csv')
#data.head()

# Getting the values and plotting it
names = ['LeftStim','RightStim','Reward','Choice','PrevChoice', 'Difficulty','Movement']
variables = []
for name in names:
    variables.append(data[name].values)

x_factor = 'LeftStim'
y_factor = 'RightStim'


x_val = data[x_factor].values #x_value
y_val = data[y_factor].values #y_value
X = np.array(list(zip(x_val, y_val))) #basically breaking down the data
#X = np.array(variables)

#Stuff for the scatter plot
plt.scatter(x_val, y_val, c='black', s=7)
plt.title("Relative Contributions from " + y_factor + " and " + x_factor)
"""behavioural variables that we currently have
#factors = ['LeftStim','RightStim','Reward','Choice','PrevChoice', 'Difficulty','Movement']

1) factor analysis
2) pca

"""
plt.xlabel(x_factor)
plt.ylabel(y_factor)

from sklearn.cluster import KMeans

k = 6 #number of clusters
kmeans = KMeans(n_clusters=k, init='k-means++')

# Fitting the input data
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
# print(X[:, 0]) #this is choice, x-axis
# print()
# print(X[:, 1]) #this is reward, y-axis
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
#algorithm starts with multple starting guesses
centroids = kmeans.cluster_centers_ #centroid values
print('centroids', centroids)
#plotting the centers
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha = 0.5)
#print("Centroids", centroids) 
plt.show()