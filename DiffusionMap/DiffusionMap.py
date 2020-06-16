import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
from scipy.spatial import distance_matrix
import sklearn.manifold as manifold
from scipy.linalg import eigh
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.utils.plot import plot_pairwise_eigenvector
from datafold.dynfold import LocalRegressionSelection


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def get_periodic_point(k, j, N):
    t_k = 2 * math.pi * (k + 1) / (N + 1)

    if j == 0:
        point = math.cos(t_k)
    else:
        point = math.sin(t_k)

    return point


def create_periodic_data(N):
    f = lambda i, j: get_periodic_point(i, j, N)
    data = np.fromfunction(np.vectorize(f), (N, 2))

    return data


def compute_diffusion_map(N, l):
    #dist = lambda p1, p2: math.sqrt(((p1 - p2) ** 2).sum())

    #D = np.asarray([[dist(p1, p2) for p2 in N] for p1 in N])

    D = distance_matrix(N,N)

    epsilon = D.max() / 20

    print(epsilon)
    W = np.exp(-(D ** 2) / epsilon)

    P = np.diag(np.sum(W, axis=1))

    inv_P = np.linalg.inv(P)

    K = inv_P @ W @ inv_P

    Q = np.diag(np.sum(K, axis=1))

    inv_Q = np.linalg.inv(Q)

    T_head = (inv_Q ** 0.5) @ K @ (inv_Q ** 0.5)

    a, v = eigh(T_head)

    # l_eigen_values = sorted(zip(a, v), reverse=True)[:l]
    #
    # a = [t[0] for t in l_eigen_values]
    # v = [list(t[1]) for t in l_eigen_values]

    a = a[-l:]
    v = v[:,-l:]

    eigen_vectors = []

    eigen_values = np.sqrt(np.power(a, 1/epsilon))

    for k in range(0, l):
        eigen_vectors.append((inv_Q ** 0.5) @ v[:,k])

    return eigen_values, eigen_vectors


def plot_periodic_data(N, l, i):

    data = create_periodic_data(N)

    values, vectors = compute_diffusion_map(data, l)

    t_k = []

    for k in range(1, N + 1):
        t_k.append(2 * math.pi * k / (N + 1))

    # Plot periodic data versus eigenvector values

    # plt.scatter(t_k, vectors[5], s=2)
    plt.plot(t_k, vectors[i], linewidth=1)

    plt.xlabel('Time')
    plt.ylabel('Eigenfunction Value')
    plt.ylim(-10, 15)
    plt.savefig('T1_f1.png')
    plt.show()

def plot_swiss_roll(n_samples):

    X, _ = make_swiss_roll(n_samples)

    values, vectors = compute_diffusion_map(X, 11)

    fig, axs = plt.subplots(3, 3,figsize=(15, 15))

    axs[0, 0].scatter(vectors[9], vectors[10],s=2)
    axs[0, 1].scatter(vectors[9], vectors[8], s=2)
    axs[0, 2].scatter(vectors[9], vectors[7], s=2)
    axs[1, 0].scatter(vectors[9], vectors[6], s=2)
    axs[1, 1].scatter(vectors[9], vectors[5], s=2)
    axs[1, 2].scatter(vectors[9], vectors[4], s=2)
    axs[2, 0].scatter(vectors[9], vectors[3], s=2)
    axs[2, 1].scatter(vectors[9], vectors[2], s=2)
    axs[2, 2].scatter(vectors[9], vectors[1], s=2)

    plt.savefig('T2_all.png')
    plt.show()


def pca_swiss(n_samples):

    X, color = make_swiss_roll(n_samples)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

    plt.savefig('T2_swiss_3D_1000.png')
    plt.show()

    pca = PCA(n_components=2, svd_solver='full')
    reducted = pca.fit_transform(X)

    fig = plt.figure()

    plt.scatter(reducted[:,0], reducted[:,1], c=color, cmap=plt.cm.Spectral)
    plt.savefig('T2_swiss_2D_1000.png')
    plt.show()


def analyze_vadere_data():
    dataset = np.loadtxt("data_DMAP_PCA_vadere.txt")

    values, vectors = compute_diffusion_map(dataset, 11)

    fig, axs = plt.subplots(3, 3,figsize=(10, 10))

    axs[0, 0].scatter(vectors[9], vectors[10], s=2)
    axs[0, 1].scatter(vectors[9], vectors[8], s=2)
    axs[0, 2].scatter(vectors[9], vectors[7], s=2)
    axs[1, 0].scatter(vectors[9], vectors[6], s=2)
    axs[1, 1].scatter(vectors[9], vectors[5], s=2)
    axs[1, 2].scatter(vectors[9], vectors[4], s=2)
    axs[2, 0].scatter(vectors[9], vectors[3], s=2)
    axs[2, 1].scatter(vectors[9], vectors[2], s=2)
    axs[2, 2].scatter(vectors[9], vectors[1], s=2)

    plt.savefig('T3_vadere.png')
    plt.show()

def datafold_swiss_roll(n_samples):

    X, color = make_swiss_roll(n_samples)

    X_pcm = pfold.PCManifold(X)
    X_pcm.optimize_parameters(result_scaling=0.5)

    print(f'epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}')

    dmap = dfold.DiffusionMaps(kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon), n_eigenpairs=9,
                               dist_kwargs=dict(cut_off=X_pcm.cut_off))
    dmap = dmap.fit(X_pcm)
    evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

    plot_pairwise_eigenvector(eigenvectors=dmap.eigenvectors_, n=1,
                              fig_params=dict(figsize=[15, 15]),
                              scatter_params=dict(cmap=plt.cm.Spectral, c=color))

    plt.savefig(f'T3_datafold_lib_{n_samples}.png')
    plt.show()


plot_swiss_roll(5000)
#plot_periodic_data(1000, 6, 5)

#pca_swiss(1000)
#analyze_vadere_data()

#datafold_swiss_roll(5000)