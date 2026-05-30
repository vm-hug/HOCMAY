# ==========================================================
# LAB 6A - Clustering Techniques
# K-Means, Mean Shift, DBSCAN
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import (
    make_blobs,
    make_moons
)

from sklearn.cluster import (
    DBSCAN,
    MeanShift
)

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

# ==========================================================
# PART A : K-MEANS FROM SCRATCH
# ==========================================================

print("=" * 60)
print("PART A - KMEANS FROM SCRATCH")
print("=" * 60)


class KMeansScratch:

    def __init__(self, k=3, max_iters=100):

        self.k = k
        self.max_iters = max_iters

    def fit(self, X):

        n_samples = X.shape[0]

        random_idx = np.random.choice(
            n_samples,
            self.k,
            replace=False
        )

        self.centroids = X[random_idx]

        for _ in range(self.max_iters):

            clusters = []

            for x in X:

                distances = np.linalg.norm(
                    x - self.centroids,
                    axis=1
                )

                cluster = np.argmin(distances)

                clusters.append(cluster)

            clusters = np.array(clusters)

            new_centroids = []

            for i in range(self.k):

                points = X[clusters == i]

                centroid = points.mean(axis=0)

                new_centroids.append(centroid)

            new_centroids = np.array(new_centroids)

            if np.allclose(
                    self.centroids,
                    new_centroids
            ):
                break

            self.centroids = new_centroids

        self.labels_ = clusters


# ==========================================================
# STEP 1 : TEST KMEANS SCRATCH
# ==========================================================

X_blob, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

model = KMeansScratch(k=4)

model.fit(X_blob)

plt.figure(figsize=(7,5))

plt.scatter(
    X_blob[:,0],
    X_blob[:,1],
    c=model.labels_,
    cmap='viridis'
)

plt.scatter(
    model.centroids[:,0],
    model.centroids[:,1],
    s=300,
    c='red'
)

plt.title("K-Means From Scratch")

plt.show()

# ==========================================================
# STEP 2 : KMEANS++
# ==========================================================

print("\nSTEP 2 - KMeans++")

kmeans_pp = KMeans(
    n_clusters=4,
    init='k-means++',
    random_state=42
)

kmeans_pp.fit(X_blob)

plt.figure(figsize=(7,5))

plt.scatter(
    X_blob[:,0],
    X_blob[:,1],
    c=kmeans_pp.labels_,
    cmap='viridis'
)

plt.scatter(
    kmeans_pp.cluster_centers_[:,0],
    kmeans_pp.cluster_centers_[:,1],
    c='red',
    s=300
)

plt.title("K-Means++")

plt.show()

# ==========================================================
# STEP 3 : ELBOW METHOD
# ==========================================================

print("\nSTEP 3 - ELBOW METHOD")

inertias = []

K_range = range(1, 11)

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(X_blob)

    inertias.append(
        model.inertia_
    )

plt.figure(figsize=(7,5))

plt.plot(
    K_range,
    inertias,
    marker='o'
)

plt.xlabel("K")
plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.show()

# ==========================================================
# STEP 4 : SILHOUETTE SCORE
# ==========================================================

print("\nSTEP 4 - SILHOUETTE SCORE")

scores = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    labels = model.fit_predict(X_blob)

    score = silhouette_score(
        X_blob,
        labels
    )

    scores.append(score)

    print(
        f"K={k} -> Silhouette={score:.4f}"
    )

plt.figure(figsize=(7,5))

plt.plot(
    range(2,11),
    scores,
    marker='o'
)

plt.xlabel("K")
plt.ylabel("Silhouette Score")

plt.title("Silhouette Analysis")

plt.show()

# ==========================================================
# PART B : CUSTOMER SEGMENTATION
# ==========================================================

print("\n")
print("=" * 60)
print("PART B - CUSTOMER SEGMENTATION")
print("=" * 60)

customers, _ = make_blobs(

    n_samples=500,

    centers=5,

    cluster_std=1.8,

    random_state=42
)

customer_kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

customer_labels = customer_kmeans.fit_predict(
    customers
)

plt.figure(figsize=(8,6))

plt.scatter(
    customers[:,0],
    customers[:,1],
    c=customer_labels,
    cmap='rainbow'
)

plt.scatter(
    customer_kmeans.cluster_centers_[:,0],
    customer_kmeans.cluster_centers_[:,1],
    s=300,
    c='black'
)

plt.title(
    "Customer Segmentation using KMeans"
)

plt.show()

# ==========================================================
# PART C : DBSCAN COMPARISON
# ==========================================================

print("\n")
print("=" * 60)
print("PART C - DBSCAN VS KMEANS")
print("=" * 60)

X_moon, y_moon = make_moons(
    n_samples=500,
    noise=0.08,
    random_state=42
)

# ----------------------------------------------------------
# KMEANS FAIL
# ----------------------------------------------------------

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

labels_kmeans = kmeans.fit_predict(
    X_moon
)

plt.figure(figsize=(7,5))

plt.scatter(
    X_moon[:,0],
    X_moon[:,1],
    c=labels_kmeans,
    cmap='coolwarm'
)

plt.title(
    "KMeans on make_moons"
)

plt.show()

# ----------------------------------------------------------
# DBSCAN SUCCESS
# ----------------------------------------------------------

dbscan = DBSCAN(
    eps=0.2,
    min_samples=5
)

labels_dbscan = dbscan.fit_predict(
    X_moon
)

plt.figure(figsize=(7,5))

plt.scatter(
    X_moon[:,0],
    X_moon[:,1],
    c=labels_dbscan,
    cmap='coolwarm'
)

plt.title(
    "DBSCAN on make_moons"
)

plt.show()

# ==========================================================
# BONUS : MEAN SHIFT
# ==========================================================

print("\nBONUS - MEAN SHIFT")

meanshift = MeanShift()

labels_ms = meanshift.fit_predict(
    X_blob
)

plt.figure(figsize=(7,5))

plt.scatter(
    X_blob[:,0],
    X_blob[:,1],
    c=labels_ms,
    cmap='viridis'
)

plt.title("Mean Shift Clustering")

plt.show()

print("\nLAB 6A COMPLETED SUCCESSFULLY")