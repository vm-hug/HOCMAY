# ==========================================================
# LAB 6B - Gaussian Mixture Models (GMM)
# Anomaly Detection using EM Algorithm
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================================
# STEP 1 : GENERATE DATASET
# ==========================================================

print("=" * 60)
print("STEP 1 - DATA GENERATION")
print("=" * 60)

# Normal traffic

X_normal, _ = make_blobs(
    n_samples=950,
    centers=1,
    cluster_std=1.2,
    random_state=42
)

# Attack traffic

X_attack, _ = make_blobs(
    n_samples=50,
    centers=[[8,8]],
    cluster_std=0.8,
    random_state=42
)

X = np.vstack([
    X_normal,
    X_attack
])

y_true = np.hstack([
    np.zeros(len(X_normal)),
    np.ones(len(X_attack))
])

print("Total Samples:", len(X))

# ==========================================================
# VISUALIZATION
# ==========================================================

plt.figure(figsize=(7,6))

plt.scatter(
    X[:,0],
    X[:,1],
    c=y_true,
    cmap='coolwarm'
)

plt.title("Ground Truth")
plt.show()

# ==========================================================
# STEP 2 : GMM FROM SCRATCH (EM)
# ==========================================================

print("\nSTEP 2 - TRAIN GMM")

class GMM:

    def __init__(
        self,
        n_components=2,
        max_iter=100
    ):
        self.k = n_components
        self.max_iter = max_iter

    def fit(self, X):

        n, d = X.shape

        np.random.seed(42)

        idx = np.random.choice(
            n,
            self.k,
            replace=False
        )

        self.means = X[idx]

        self.covs = [
            np.eye(d)
            for _ in range(self.k)
        ]

        self.weights = np.ones(
            self.k
        ) / self.k

        for _ in range(self.max_iter):

            responsibilities = np.zeros(
                (n, self.k)
            )

            # E-Step

            for k in range(self.k):

                responsibilities[:,k] = (
                    self.weights[k]
                    *
                    multivariate_normal.pdf(
                        X,
                        mean=self.means[k],
                        cov=self.covs[k]
                    )
                )

            responsibilities /= (
                responsibilities.sum(
                    axis=1,
                    keepdims=True
                )
            )

            # M-Step

            Nk = responsibilities.sum(
                axis=0
            )

            for k in range(self.k):

                self.means[k] = (
                    responsibilities[:,k][:,None]
                    * X
                ).sum(axis=0) / Nk[k]

                diff = X - self.means[k]

                self.covs[k] = (
                    responsibilities[:,k][:,None,None]
                    *
                    np.einsum(
                        'ni,nj->nij',
                        diff,
                        diff
                    )
                ).sum(axis=0) / Nk[k]

                self.covs[k] += (
                    1e-6 *
                    np.eye(d)
                )

            self.weights = Nk / n

    def score_samples(self, X):

        likelihood = np.zeros(
            (len(X), self.k)
        )

        for k in range(self.k):

            likelihood[:,k] = (
                self.weights[k]
                *
                multivariate_normal.pdf(
                    X,
                    mean=self.means[k],
                    cov=self.covs[k]
                )
            )

        return np.log(
            likelihood.sum(axis=1)
            + 1e-12
        )

# ==========================================================
# TRAIN ONLY NORMAL DATA
# ==========================================================

gmm = GMM(
    n_components=2,
    max_iter=50
)

gmm.fit(X_normal)

# ==========================================================
# STEP 3 : LOG LIKELIHOOD
# ==========================================================

print("\nSTEP 3 - LOG LIKELIHOOD")

scores = gmm.score_samples(X)

# ==========================================================
# STEP 4 : THRESHOLD
# ==========================================================

print("\nSTEP 4 - THRESHOLD")

threshold = np.percentile(
    scores,
    5
)

print("Threshold =", threshold)

# ==========================================================
# STEP 5 : DETECT ANOMALIES
# ==========================================================

print("\nSTEP 5 - ANOMALY DETECTION")

anomaly_pred = (
    scores < threshold
).astype(int)

# ==========================================================
# VISUALIZE
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    X[:,0],
    X[:,1],
    c=anomaly_pred,
    cmap='coolwarm'
)

plt.title(
    "Detected Anomalies (Red)"
)

plt.show()

# ==========================================================
# STEP 6 : EVALUATION
# ==========================================================

print("\nSTEP 6 - EVALUATION")

precision = precision_score(
    y_true,
    anomaly_pred
)

recall = recall_score(
    y_true,
    anomaly_pred
)

f1 = f1_score(
    y_true,
    anomaly_pred
)

print(
    f"Precision = {precision:.4f}"
)

print(
    f"Recall = {recall:.4f}"
)

print(
    f"F1 Score = {f1:.4f}"
)

cm = confusion_matrix(
    y_true,
    anomaly_pred
)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.show()

# ==========================================================
# STEP 7 : DISTRIBUTION OF SCORES
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(
    scores,
    bins=40
)

plt.axvline(
    threshold,
    linestyle='--'
)

plt.title(
    "Log-Likelihood Distribution"
)

plt.show()

# ==========================================================
# STEP 8 : KMEANS COMPARISON
# ==========================================================

print("\nSTEP 7 - GMM vs KMEANS")

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

labels = kmeans.fit_predict(X)

centers = kmeans.cluster_centers_

distances = np.min(
    np.linalg.norm(
        X[:,None]
        - centers,
        axis=2
    ),
    axis=1
)

threshold_kmeans = np.percentile(
    distances,
    95
)

anomaly_kmeans = (
    distances > threshold_kmeans
).astype(int)

precision_k = precision_score(
    y_true,
    anomaly_kmeans
)

recall_k = recall_score(
    y_true,
    anomaly_kmeans
)

f1_k = f1_score(
    y_true,
    anomaly_kmeans
)

comparison = pd.DataFrame({

    "Method":[
        "GMM",
        "KMeans"
    ],

    "Precision":[
        precision,
        precision_k
    ],

    "Recall":[
        recall,
        recall_k
    ],

    "F1":[
        f1,
        f1_k
    ]
})

print("\nComparison Table")
print(comparison)

# ==========================================================
# FINAL
# ==========================================================

print("\nLAB 6B COMPLETED SUCCESSFULLY")