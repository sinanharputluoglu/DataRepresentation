import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

x = np.loadtxt("res/pca_dataset.txt")
pca = PCA(n_components=2, whiten=True, svd_solver='full')
pca.fit(x)
print(pca.explained_variance_)
print(sum(pca.explained_variance_))
print(pca.components_)



fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# plot data
ax.scatter(x[:, 0], x[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v, ax=ax)

ax.axis('equal')
ax.set(xlabel='x', ylabel='f(x)')
ax.set_xlim(-3,3)



fig.savefig('figures/pca_direction.png')




