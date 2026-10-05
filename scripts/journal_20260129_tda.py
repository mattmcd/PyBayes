# %%
# Try out persistent homology on the sklearn plot_cluster_comparison example

# %%
from startup import np, pd, plt, sns
# %%
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.diagrams import PersistenceEntropy

# %%
import networkx as nx

# %%
# Code from https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
import time
import warnings
from itertools import cycle, islice

# import matplotlib.pyplot as plt
# import numpy as np

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# %%
df = pd.DataFrame(noisy_circles[0], columns=['x', 'y'])
sns.scatterplot(df, x='x', y='y')
plt.show()

# %%
VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])  # Parameter explained in the text
point_clouds = df.values[None, ...]
print(f"There are {point_clouds.shape[0]} point clouds in {point_clouds.shape[2]} dimensions, "
      f"each with {point_clouds.shape[1]} points.")
diagrams = VR.fit_transform(point_clouds)
print(diagrams.shape)

# %%

PE = PersistenceEntropy()
features = PE.fit_transform(diagrams)
print(features.shape)
# %%
plot_diagram(diagrams[0])
# plt.show()

# %%
G = nx.Graph()

for i, row in df.iterrows():
    G.add_node(i, pos=(row['x'], row['y']))

threshold = 0.14
edges_to_add_list = nx.geometric_edges(G, threshold, p = 2)
G.add_edges_from(edges_to_add_list)

cc = list(nx.connected_components(G))
cc_size = [len(x) for x in cc]
print(f'Number of connected components: {len(cc)}')
print(f'Sizes: {cc_size}')

# %%
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
df.plot.scatter(x='x', y='y', ax=axs[0])
nx.draw(G, node_size=1, ax=axs[1], pos=df.values)
plt.show()