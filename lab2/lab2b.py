import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import umap
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST
(X_train, y_train), (_, _) = fashion_mnist.load_data()

# Lấy 5000 mẫu
n_samples = 5000
X = X_train[:n_samples]
y = y_train[:n_samples]

# Flatten: (5000, 28, 28) → (5000, 784)
X = X.reshape(n_samples, -1)

# Normalize
X = X / 255.0

start = time.time()

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

time_pca = time.time() - start


start = time.time()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='random',
    random_state=42
)
X_tsne = tsne.fit_transform(X)

time_tsne = time.time() - start


start = time.time()

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_umap = umap_model.fit_transform(X)

time_umap = time.time() - start


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=5)
axes[0].set_title("PCA (2D)")

scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=5)
axes[1].set_title("t-SNE (perplexity=30)")

scatter3 = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=5)
axes[2].set_title("UMAP (n_neighbors=15)")

plt.colorbar(scatter3, ax=axes, fraction=0.02)
plt.tight_layout()
plt.show()




