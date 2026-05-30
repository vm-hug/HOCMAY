# ==========================================================
# LAB 4C - Classification Techniques (SVM)
# ==========================================================
# Mục tiêu:
# 1. Maximum Margin và Support Vectors
# 2. Hard Margin vs Soft Margin
# 3. Kernel Trick (RBF Kernel)
# 4. Ảnh hưởng của C và gamma
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.svm import SVC

# ==========================================================
# HÀM VẼ DECISION BOUNDARY
# ==========================================================

def plot_decision_boundary(model, X, y, title):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors='k'
    )

    plt.title(title)

# ==========================================================
# BƯỚC 1: TẠO DỮ LIỆU MAKE_MOONS
# ==========================================================

print("=" * 50)
print("STEP 1: MAKE_MOONS DATASET")
print("=" * 50)

X, y = make_moons(
    n_samples=300,
    noise=0.2,
    random_state=42
)

plt.figure(figsize=(6, 5))
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    edgecolors='k'
)

plt.title("make_moons Dataset")
plt.show()

# ==========================================================
# BƯỚC 2: LINEAR SVM
# ==========================================================

print("=" * 50)
print("STEP 2: LINEAR SVM")
print("=" * 50)

linear_svm = SVC(
    kernel='linear',
    C=1
)

linear_svm.fit(X, y)

plt.figure(figsize=(7, 5))
plot_decision_boundary(
    linear_svm,
    X,
    y,
    "Linear SVM"
)

plt.show()

# ==========================================================
# BƯỚC 3: RBF KERNEL
# ==========================================================

print("=" * 50)
print("STEP 3: RBF KERNEL")
print("=" * 50)

rbf_svm = SVC(
    kernel='rbf',
    C=1,
    gamma=1
)

rbf_svm.fit(X, y)

plt.figure(figsize=(7, 5))
plot_decision_boundary(
    rbf_svm,
    X,
    y,
    "RBF Kernel SVM"
)

plt.show()

# ==========================================================
# BƯỚC 4: THỬ NGHIỆM C
# ==========================================================

print("=" * 50)
print("STEP 4: EFFECT OF C")
print("=" * 50)

C_values = [0.1, 1, 10, 100]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

for ax, C in zip(axes.ravel(), C_values):

    model = SVC(
        kernel='rbf',
        C=C,
        gamma=1
    )

    model.fit(X, y)

    plt.sca(ax)

    plot_decision_boundary(
        model,
        X,
        y,
        f"C = {C}"
    )

plt.tight_layout()
plt.show()

# ==========================================================
# BƯỚC 5: THỬ NGHIỆM GAMMA
# ==========================================================

print("=" * 50)
print("STEP 5: EFFECT OF GAMMA")
print("=" * 50)

gamma_values = [0.1, 1, 10]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

for ax, gamma in zip(axes, gamma_values):

    model = SVC(
        kernel='rbf',
        C=1,
        gamma=gamma
    )

    model.fit(X, y)

    plt.sca(ax)

    plot_decision_boundary(
        model,
        X,
        y,
        f"gamma = {gamma}"
    )

plt.tight_layout()
plt.show()

# ==========================================================
# BƯỚC 6: SUPPORT VECTORS
# ==========================================================

print("=" * 50)
print("STEP 6: SUPPORT VECTORS")
print("=" * 50)

model_sv = SVC(
    kernel='rbf',
    C=1,
    gamma=1
)

model_sv.fit(X, y)

plt.figure(figsize=(8, 6))

plot_decision_boundary(
    model_sv,
    X,
    y,
    "Support Vectors"
)

# Highlight Support Vectors

plt.scatter(
    model_sv.support_vectors_[:, 0],
    model_sv.support_vectors_[:, 1],
    s=200,
    facecolors='none',
    edgecolors='red',
    linewidths=2,
    label='Support Vectors'
)

plt.legend()

plt.show()

# ==========================================================
# THÔNG TIN SUPPORT VECTOR
# ==========================================================

print("\nNumber of Support Vectors:")
print(len(model_sv.support_vectors_))

print("\nSupport Vectors Coordinates:")
print(model_sv.support_vectors_)

print("\nLAB 4C COMPLETED SUCCESSFULLY")