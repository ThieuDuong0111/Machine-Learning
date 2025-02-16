import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Tạo tập dữ liệu mẫu
data = {
    "Annual_Income": [15, 16, 17, 18, 30, 31, 32, 33, 50, 51, 52, 53, 70, 71, 72, 73],
    "Spending_Score": [80, 85, 88, 90, 50, 52, 48, 49, 40, 42, 43, 45, 20, 22, 18, 19]
}

df = pd.DataFrame(data)

# Hiển thị dữ liệu
print(df.head())

# Chuyển DataFrame thành numpy array
X = df.values  

# Vẽ Dendrogram
plt.figure(figsize=(10, 5))
sch.dendrogram(sch.linkage(X, method="ward"))
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.title("Dendrogram để tìm số cụm tối ưu")
plt.show()

# Khởi tạo mô hình với 3 cụm
hc = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
df["Cluster"] = hc.fit_predict(X)

# Hiển thị kết quả
print(df)

# Vẽ scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=df["Cluster"], cmap="viridis", edgecolors="k", s=100)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Hierarchical Clustering (3 Clusters)")
plt.show()
