# =====================================================
# LAB 3B
# Polynomial Regression & Logistic Regression
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix

# =====================================================
# PART A : POLYNOMIAL REGRESSION
# =====================================================

print("=" * 50)
print("PART A - POLYNOMIAL REGRESSION")
print("=" * 50)

# Tạo dữ liệu phi tuyến

np.random.seed(42)

X = np.random.uniform(-3, 3, 100)
y = X**2 + np.random.normal(0, 1, 100)

X = X.reshape(-1, 1)

# -----------------------------------------------------
# Hiển thị dữ liệu
# -----------------------------------------------------

plt.figure(figsize=(6,4))
plt.scatter(X, y)
plt.title("Non-linear Dataset")
plt.show()

# -----------------------------------------------------
# Polynomial Regression
# -----------------------------------------------------

degrees = [1, 2, 4, 10, 15]

for degree in degrees:

    poly = PolynomialFeatures(degree)

    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    x_plot_poly = poly.transform(x_plot)

    y_plot = model.predict(x_plot_poly)

    plt.figure(figsize=(6,4))
    plt.scatter(X, y, color='blue')
    plt.plot(x_plot, y_plot, color='red')

    plt.title(f"Polynomial Degree = {degree}")
    plt.show()

# -----------------------------------------------------
# Learning Curves
# -----------------------------------------------------

for degree in [1, 2, 15]:

    poly = PolynomialFeatures(degree)

    X_poly = poly.fit_transform(X)

    model = LinearRegression()

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_poly,
        y,
        cv=5,
        scoring='neg_mean_squared_error'
    )

    train_error = -train_scores.mean(axis=1)
    val_error = -val_scores.mean(axis=1)

    plt.figure(figsize=(6,4))

    plt.plot(
        train_sizes,
        train_error,
        marker='o',
        label='Train Error'
    )

    plt.plot(
        train_sizes,
        val_error,
        marker='o',
        label='Validation Error'
    )

    plt.title(f"Learning Curve (Degree={degree})")
    plt.xlabel("Training Examples")
    plt.ylabel("MSE")
    plt.legend()

    plt.show()

# =====================================================
# PART B : LOGISTIC REGRESSION
# =====================================================

print("\n")
print("=" * 50)
print("PART B - LOGISTIC REGRESSION")
print("=" * 50)

# -----------------------------------------------------
# Load Iris Dataset
# -----------------------------------------------------

iris = load_iris()

X = iris.data[:, :2]

# Binary Classification
y = (iris.target == 0).astype(int)

print("Dataset Shape:", X.shape)

# -----------------------------------------------------
# EDA
# -----------------------------------------------------

plt.figure(figsize=(6,4))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap='bwr'
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset")

plt.show()

# -----------------------------------------------------
# Logistic Regression from Scratch
# -----------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegressionGD:

    def __init__(self, lr=0.1, epochs=5000):

        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):

        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        self.losses = []

        for _ in range(self.epochs):

            z = np.dot(X, self.w) + self.b

            y_hat = sigmoid(z)

            loss = -np.mean(
                y * np.log(y_hat + 1e-8)
                + (1 - y) * np.log(1 - y_hat + 1e-8)
            )

            self.losses.append(loss)

            dw = (1 / m) * np.dot(X.T, (y_hat - y))
            db = (1 / m) * np.sum(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):

        z = np.dot(X, self.w) + self.b

        return sigmoid(z)

    def predict(self, X):

        probs = self.predict_proba(X)

        return (probs >= 0.5).astype(int)


# -----------------------------------------------------
# Train/Test Split
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------------------
# Train Model
# -----------------------------------------------------

model = LogisticRegressionGD(
    lr=0.1,
    epochs=5000
)

model.fit(X_train, y_train)

# -----------------------------------------------------
# Loss Curve
# -----------------------------------------------------

plt.figure(figsize=(6,4))

plt.plot(model.losses)

plt.title("Log Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy =", accuracy)

# -----------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5,4))

plt.imshow(cm)

plt.title("Confusion Matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j])

plt.colorbar()
plt.show()

# -----------------------------------------------------
# Decision Boundary
# -----------------------------------------------------

x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1

y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(grid)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(6,4))

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
    cmap='bwr'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary")

plt.show()

print("\nLab 3B Completed Successfully!")