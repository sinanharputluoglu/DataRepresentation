from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.misc import face
from skimage.transform import resize

picture = resize(face(gray=True), output_shape=(249, 185))

components_array = [120, 50, 10]

fig1,ax1 = plt.subplots(1,4)
fig1.subplots_adjust(wspace=0.6)
fig2,ax2 = plt.subplots()

for i in range(3):
    pca = PCA(n_components= components_array[i], svd_solver='full')
    pca.fit(picture)
    components = pca.transform(picture)
    projected = pca.inverse_transform(components)

    ax1[i].imshow(projected)
    ax1[i].set_xlabel(components_array[i].__str__() +'-dim\nreconstruction')
    print(sum(pca.explained_variance_))


pca = PCA(svd_solver='full')
pca.fit(picture)
components = pca.transform(picture)
projected = pca.inverse_transform(components)
ax1[3].imshow(projected)
ax1[3].set_xlabel('all components\nreconstruction')
print(sum(pca.explained_variance_))



ax2.imshow(picture)
ax2.set_ylabel('full-dim\ninput')


fig1.savefig('figures/part2.png')
fig2.savefig('figures/part2_initial.png')


