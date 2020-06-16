import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#prepare dataset
dataset = np.loadtxt("res/data_DMAP_PCA_vadere.txt")

x1 = dataset[:, 0]
y1 = dataset[:, 1]
x2 = dataset[:, 2]
y2 = dataset[:, 3]

fig, ax = plt.subplots(1, 2, figsize=(8, 6))

# plot data
ax[0].scatter(x1, y1, alpha=0.2)
ax[1].scatter(x2, y2, alpha=0.2)

ax[0].axis('equal')
ax[0].set(xlabel='x1', ylabel='y1')
ax[1].axis('equal')
ax[1].set(xlabel='x2', ylabel='y2')

fig.savefig('figures/part3_visualizepaths.png')


pca = PCA(svd_solver='full', n_components=2)
pca.fit(dataset)
print(sum(pca.explained_variance_))
print(pca.explained_variance_)

