import numpy as np
import pandas as dp
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000

ad_spend = np.random.uniform(20 , 150 , n)
noise = np.random.normal(0 , 20 , n)

revenue = 3.0 * ad_spend + 10 + noise

df = dp.DataFrame({
    "Ad Spend" : ad_spend,
    "Revenue" : revenue
})

df["Performance"] = np.where(df["Revenue"] > 200 , "High" , "Low")

df_clean = df[(df["Revenue"] >= 0) & (df["Revenue"] <= 500)]

print("Số dòng dữ liệu : ", df_clean.shape[0])
print(df_clean.head())

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(df_clean["Revenue"] , bins=30, color='skyblue', edgecolor='black')
plt.title("Phân phối Doanh Thu")
plt.xlabel("Revenue")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
colors = df_clean["Performance"].map({
    "High" : "green",
    "Low" : "red"
})
plt.scatter(df_clean["Ad Spend"], df_clean["Revenue"], c=colors, alpha=0.6)
plt.title("Ad Spend vs Revenue")
plt.xlabel("Ad Spend")
plt.ylabel("Revenue")

plt.tight_layout()
plt.show()