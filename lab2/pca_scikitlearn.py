import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

wines = pd.read_csv('WineQT.csv')
wines.head()

X = wines.drop(['quality', 'Id'], axis=1)
y = wines['quality']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

#Tính ma trận hiệp phương sai (Covariance Matrix)
cov_mat = np.cov(X_std.T)

#Phân rã Trị riêng (Eigenvalues) & Vector riêng (Eigenvectors)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

#Sắp xếp giảm dần theo trị riêng
sorted_index = np.argsort(eigen_vals)[::-1]
eigen_vals_sorted = eigen_vals[sorted_index]
eigen_vecs_sorted = eigen_vecs[:, sorted_index]

#Tính phương sai giải thích
tot = sum(eigen_vals_sorted)
var_exp_manual = [(i / tot) for i in eigen_vals_sorted]
cum_var_exp_manual = np.cumsum(var_exp_manual)

#Lấy 2 vector riêng đầu tiên làm ma trận trọng số W
W = eigen_vecs_sorted[:, :2]
X_pca_manual = X_std.dot(W)

print("Manual PCA shape:", X_pca_manual.shape)

pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X_std)
var_exp_sklearn = pca_sklearn.explained_variance_ratio_

# 4. Chuẩn bị dữ liệu so sánh
comparison_data = {
    'Component': ['PC1', 'PC2', 'PC1', 'PC2'],
    'Method': ['Manual (Numpy)', 'Manual (Numpy)', 'Scikit-Learn', 'Scikit-Learn'],
    'Variance Ratio': [var_exp_manual[0], var_exp_manual[1],
                       var_exp_sklearn[0], var_exp_sklearn[1]]
}
comp_df = pd.DataFrame(comparison_data)

# 5. Vẽ biểu đồ với Seaborn
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Vẽ barplot nhóm theo Method
ax = sns.barplot(data=comp_df, x='Component', y='Variance Ratio', hue='Method', palette='Set2')

# Trang trí
plt.title('So sánh Tỷ lệ Phương sai: Manual vs Scikit-Learn', fontsize=16)
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)

# Thêm nhãn số liệu trên đầu cột
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', padding=3)

plt.tight_layout()
plt.show()

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X_std)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Scree Plot
axes[0, 0].bar(range(1, len(var_exp_manual) + 1), var_exp_manual, alpha=0.5, label='Individual variance')
axes[0, 0].step(range(1, len(cum_var_exp_manual) + 1), cum_var_exp_manual, where='mid', label='Cumulative variance')
axes[0, 0].set_ylabel('Explained variance ratio')
axes[0, 0].set_xlabel('Principal component index')
axes[0, 0].set_title('Scree Plot (Manual)')
axes[0, 0].legend(loc='best')

# 2. Manual PCA Plot
sns.scatterplot(x=X_pca_manual[:, 0], y=X_pca_manual[:, 1], hue=y, palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title('Manual PCA (Numpy)')

# 3. Sklearn PCA Plot
sns.scatterplot(x=X_pca_sklearn[:, 0], y=X_pca_sklearn[:, 1], hue=y, palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title('Scikit-Learn PCA')

# 4. Kernel PCA Plot
sns.scatterplot(x=X_kpca[:, 0], y=X_kpca[:, 1], hue=y, palette='Set2', ax=axes[1, 1])
axes[1, 1].set_title('Kernel PCA (RBF)')

plt.tight_layout()
plt.show()