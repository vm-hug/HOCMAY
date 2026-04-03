import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from collections import Counter

# 1. Load dữ liệu trực tiếp từ link raw (để chạy thẳng trên Colab)
url = 'https://raw.githubusercontent.com/shivang98/Social-Network-ads-Boost/master/Social_Network_Ads.csv'
df = pd.read_csv(url)


# Lấy 2 đặc trưng: Age và EstimatedSalary
X = df[['Age', 'EstimatedSalary']].values
# Lấy nhãn: Purchased (0: Không mua, 1: Có mua)
y = df['Purchased'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trực quan hóa dữ liệu gốc
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolor='k', s=40)
plt.title("Tập dữ liệu huấn luyện (Original Scale)")
plt.xlabel("Độ tuổi (Age)")
plt.ylabel("Thu nhập ước tính (Estimated Salary)")
plt.legend(*scatter.legend_elements(), title="Purchased")
plt.show()

from sklearn.preprocessing import StandardScaler
from collections import Counter
import numpy as np

# 1. Feature Scaling (Bắt buộc cho KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Xây dựng Class CustomKNN nâng cao
class CustomKNN:
    def __init__(self, n_neighbors=5, p=2, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _minkowski_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def predict(self, X):
        X = np.array(X)
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        distances = [self._minkowski_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        if self.weights == 'uniform':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]

        elif self.weights == 'distance':
            class_weights = {}
            for i in range(self.n_neighbors):
                label = k_nearest_labels[i]
                dist = k_nearest_distances[i]
                if dist == 0:
                    return label
                weight = 1.0 / dist
                class_weights[label] = class_weights.get(label, 0) + weight
            return max(class_weights, key=class_weights.get)
        else:
            raise ValueError("Tham số weights chỉ nhận 'uniform' hoặc 'distance'")

# Đánh giá thử mô hình với uniform vs distance weights
knn_uniform = CustomKNN(n_neighbors=5, p=2, weights='uniform')
knn_uniform.fit(X_train_scaled, y_train)
acc_uniform = np.sum(knn_uniform.predict(X_test_scaled) == y_test) / len(y_test)

knn_distance = CustomKNN(n_neighbors=5, p=2, weights='distance')
knn_distance.fit(X_train_scaled, y_train)
acc_distance = np.sum(knn_distance.predict(X_test_scaled) == y_test) / len(y_test)

print(f"Độ chính xác (Uniform Weights): {acc_uniform * 100:.2f}%")
print(f"Độ chính xác (Distance Weights): {acc_distance * 100:.2f}%")


# Hàm vẽ Decision Boundary sử dụng CustomKNN
def plot_decision_boundary(X, y, k_values):
    # Bước nhảy lưới (step size) - Để h=0.1 để cân bằng giữa độ mịn và tốc độ chạy
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF']) # Đỏ nhạt, Xanh nhạt
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])   # Đỏ đậm, Xanh đậm

    for idx, k in enumerate(k_values):
        # Khởi tạo mô hình CustomKNN (Dùng weights='uniform' làm mặc định để vẽ)
        clf = CustomKNN(n_neighbors=k, p=2, weights='uniform')
        clf.fit(X, y)

        # Dự đoán cho mọi điểm trên lưới
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[idx].pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.6)
        # Vẽ các điểm dữ liệu train đè lên ranh giới
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        axes[idx].set_xlim(xx.min(), xx.max())
        axes[idx].set_ylim(yy.min(), yy.max())
        axes[idx].set_title(f"Decision Boundary (K = {k})")
        axes[idx].set_xlabel("Age (Scaled)")
        axes[idx].set_ylabel("Estimated Salary (Scaled)")

    plt.tight_layout()
    plt.show()

# Chạy với các K khác nhau (1, 5, 15, 50)
k_list = [1, 5, 15, 50]
print("Đang chạy dự đoán và vẽ ranh giới... (Quá trình tính toán thủ công sẽ mất một chút thời gian)")
plot_decision_boundary(X_train_scaled, y_train, k_list)

from sklearn.model_selection import KFold
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt

# Khởi tạo K-Fold với 5 phần
kf = KFold(n_splits=5, shuffle=True, random_state=42)
k_range = range(1, 40)
cv_error_rates = []

print("Đang chạy Cross-Validation để tìm K tối ưu. Vui lòng đợi...")

for k in k_range:
    fold_errors = []
    # Lặp qua từng fold
    for train_index, val_index in kf.split(X_train_scaled):
        # Chia dữ liệu theo index của K-Fold
        X_cv_train, X_cv_val = X_train_scaled[train_index], X_train_scaled[val_index]
        y_cv_train, y_cv_val = y_train[train_index], y_train[val_index]

        # Khởi tạo và train mô hình
        knn = CustomKNN(n_neighbors=k, p=2, weights='uniform')
        knn.fit(X_cv_train, y_cv_train)

        # Dự đoán và tính lỗi trên tập validation
        y_pred = knn.predict(X_cv_val)
        error = np.mean(y_pred != y_cv_val)
        fold_errors.append(error)

    # Lấy trung bình lỗi của 5 folds
    cv_error_rates.append(np.mean(fold_errors))

# Tìm K tối ưu
optimal_k = k_range[cv_error_rates.index(min(cv_error_rates))]

# KỸ THUẬT VẼ ĐƯỜNG CONG MƯỢT (SMOOTHING) DÙNG SCIPY
x_smooth = np.linspace(min(k_range), max(k_range), 300)
spline = make_interp_spline(k_range, cv_error_rates, k=3) # Nội suy bậc 3
y_smooth = spline(x_smooth)

plt.figure(figsize=(10, 6))
# Vẽ đường cong mượt
plt.plot(x_smooth, y_smooth, color='blue', linestyle='-', label='Đường cong xu hướng (Smoothed)')
# Vẽ các điểm dữ liệu thực tế
plt.plot(k_range, cv_error_rates, 'ro', label='Lỗi trung bình (CV Error)', markersize=6, alpha=0.6)

plt.title('Tỷ lệ lỗi Cross-Validation theo giá trị K (Smoothed)')
plt.xlabel('Giá trị K (Số láng giềng)')
plt.ylabel('Tỷ lệ lỗi Trung bình')
plt.axvline(x=optimal_k, color='green', linestyle='--', label=f'K tối ưu = {optimal_k}')

plt.legend()
plt.grid(True)
plt.show()

print(f"👉 Dựa vào Cross-Validation, giá trị K tối ưu nhất mang lại tỷ lệ lỗi thấp nhất là: K = {optimal_k}")