import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Mục tiêu là sử dụng K-Means Clustering để chia khách hàng thành các nhóm khác nhau.

# Tạo tập dữ liệu mẫu
data = {
    "Annual_Income": [15, 16, 17, 18, 30, 31, 32, 33, 50, 51, 52, 53, 70, 71, 72, 73],
    "Spending_Score": [80, 85, 88, 90, 50, 52, 48, 49, 40, 42, 43, 45, 20, 22, 18, 19]
}

df = pd.DataFrame(data)

# Hiển thị 5 dòng đầu
print(df.head())

# Chuẩn bị dữ liệu đầu vào (X)
X = df.values  # Chuyển DataFrame thành numpy array

# Khởi tạo mô hình K-Means với 3 cụm (clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Gán nhãn cụm cho từng điểm dữ liệu
df["Cluster"] = kmeans.labels_

# Hiển thị kết quả
print(df)

# Vẽ scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=df["Cluster"], cmap="viridis", edgecolors="k", s=100)

# Vẽ tâm cụm
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centroids")

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering (3 Clusters)")
plt.legend()
plt.show()

inertia = []
k_range = range(1, 10)  # Thử từ 1 đến 9 cụm

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)  # Lưu giá trị inertia

# Vẽ biểu đồ Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker="o", linestyle="--")
plt.xlabel("Số cụm (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method để tìm số cụm tối ưu")
plt.show()


