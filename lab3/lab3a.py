import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Bước 1 & 2: Tạo dữ liệu và Cài đặt Gradient Descent

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

#Bước 3: Thử nghiệm với các Learning Rates
learning_rates = [0.001, 0.01, 0.1, 1.0]
iterations = 500
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    theta_initial = np.random.randn(2, 1)
    _, cost_history = gradient_descent(X_b, y, theta_initial, lr, iterations)
    plt.plot(range(iterations), cost_history, label=f'LR = {lr}')

plt.xlabel('Iterations')
plt.ylabel('Loss (MSE)')
plt.title('Loss vs Iterations với các Learning Rates khác nhau')
plt.legend()
plt.ylim(0, 20) # Giới hạn để dễ quan sát
plt.show()

#Bước 4 & 5: So sánh Normal Equation và Scikit-learn
# Normal Equation
start_time = time.time()
theta_best_ne = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
ne_time = time.time() - start_time
ne_mse = mean_squared_error(y, X_b.dot(theta_best_ne))

# Gradient Descent (chọn LR tốt nhất là 0.1)
start_time = time.time()
theta_best_gd, _ = gradient_descent(X_b, y, np.random.randn(2, 1), 0.1, 1000)
gd_time = time.time() - start_time
gd_mse = mean_squared_error(y, X_b.dot(theta_best_gd))

# Sklearn
lin_reg = LinearRegression()
start_time = time.time()
lin_reg.fit(X, y)
sk_time = time.time() - start_time
sk_mse = mean_squared_error(y, lin_reg.predict(X))

plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best_gd)

plt.plot(X_new, y_predict, "r-", linewidth=2, label="Đường hồi quy (GD)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()