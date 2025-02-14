import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Tạo tập dữ liệu: [Trọng lượng, Kích thước]
X = np.array([
    [150, 7], [160, 7.5], [170, 8],  # Táo
    [180, 9], [200, 10], [210, 10.5]  # Cam
])

# Nhãn (0 = Táo, 1 = Cam)
y = np.array([0, 0, 0, 1, 1, 1])

# Tạo mô hình KNN với K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

new_fruit = np.array([[185, 9.5]])  # Trái cây mới có trọng lượng 175g và kích thước 8.5cm
prediction = knn.predict(new_fruit)

# Hiển thị kết quả
fruit_label = "Apple 🍏" if prediction[0] == 0 else "Orange 🍊"
print("Loại trái cây dự đoán:", fruit_label)

# Vẽ dữ liệu
plt.scatter(X[:3, 0], X[:3, 1], color="red", label="Apple")  # Táo
plt.scatter(X[3:, 0], X[3:, 1], color="orange", label="Orange")  # Cam

# Vẽ điểm dự đoán
plt.scatter(new_fruit[:, 0], new_fruit[:, 1], color="blue", marker="*", s=200, label="New Fruit")

plt.xlabel("Weight (g)")
plt.ylabel("Size (cm)")
plt.legend()
plt.title("K-Nearest Neighbors (KNN) Classification")
plt.show()

