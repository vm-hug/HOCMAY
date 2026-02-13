import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Tạo dữ liệu giả lập: y = 4 + 3x + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Thêm cột bias (x0 = 1) vào X
X_b = np.c_[np.ones((100, 1)), X]


def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history