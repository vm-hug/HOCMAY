from scipy.sparse import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Bước 1: THU THẬP DỮ LIỆU
prices_house = pd.read_csv('house_cleaned.csv')

# Bước 2: CHUẨN BỊ DỮ LIỆU (LÀM SẠCH & CHỌN LỌC)
selected_cols = ['price', 'area', 'bedRoom', 'bathroom', 'balcony', 'floorNum']
prices_house = prices_house[selected_cols].copy()


def clean_balcony(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        try:
            return float(re.findall(r'\d+', x)[0])
        except:
            return 0
    return float(x)


prices_house['balcony'] = prices_house['balcony'].apply(clean_balcony)

imputer = SimpleImputer(strategy='median')
data_cleaned = pd.DataFrame(imputer.fit_transform(prices_house), columns=prices_house.columns)
data_cleaned.shape

X = data_cleaned.drop('price', axis=1)
y = data_cleaned['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bước 3: KHÁM PHÁ VÀ TRỰC QUAN HÓA
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

fig.suptitle(
    'PHÂN TÍCH GIÁ NHÀ & MÔ HÌNH DỰ ĐOÁN',
    fontsize=16,
    fontweight='bold'
)

# Histogram giá nhà
sns.histplot(
    data_cleaned['price'],
    kde=True,
    color='skyblue',
    ax=ax[0]
)
ax[0].set_title('Phân phối Giá nhà')

# Heatmap tương quan
corr = data_cleaned.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    cmap='Blues',
    fmt=".2f",
    ax=ax[1],
    square=True,
    linewidths=1,
    linecolor='white',
    cbar_kws={"shrink": .5}
)
ax[1].set_title('Ma trận tương quan')

sns.scatterplot(
    x=np.log10(data_cleaned['area']),
    y=data_cleaned['price'],
    color='orange',
    alpha=0.6,
    ax=ax[2]
)
ax[2].set_title('Tương quan log(Diện tích) - Giá')
ax[2].set_xlabel('log10(area)')

plt.tight_layout()
plt.show()

# Bước 4 , 5: HUẤN LUYỆN & TINH CHỈNH
rf = RandomForestRegressor(random_state=32)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Tham số tối ưu tìm được:", grid_search.best_params_)

# Bước 6: ĐÁNH GIÁ MÔ HÌNH
y_pred = best_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R2 Score (Độ chính xác): {r2:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))

# Scatter: giá thực tế vs dự đoán
ax.scatter(
    y_test,
    y_pred,
    alpha=0.6,
    color='purple'
)

# Đường chéo y = x (dự đoán hoàn hảo)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    'k--',
    lw=2
)

ax.set_xlabel('Giá thực tế')
ax.set_ylabel('Giá dự đoán')
ax.set_title(f'Kết quả dự đoán (R2 = {r2:.2f})')
ax.grid(True)

plt.tight_layout()
plt.show()
