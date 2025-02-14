import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Tạo dữ liệu giả lập: Đặc trưng (Feature) là chiều cao, cân nặng và giới tính (0 = Nữ, 1 = Nam)
data = pd.DataFrame({
    'Height': [150, 160, 165, 170, 175, 180, 185, 190, 155, 158],
    'Weight': [50, 55, 60, 70, 80, 85, 90, 95, 52, 58],
    'Gender': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  # 0: Nữ, 1: Nam
})

# Chia thành feature X và target y
X = data[['Height', 'Weight']].values
y = data['Gender'].values

# Chia dữ liệu thành tập train và test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (StandardScaler giúp đưa dữ liệu về cùng tỉ lệ)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo mô hình SVC với kernel 'rbf' (Radial Basis Function)
svc_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # C là hệ số điều chỉnh, gamma xác định ảnh hưởng của điểm dữ liệu
svc_model.fit(X_train_scaled, y_train)  # Huấn luyện mô hình

# Dự đoán trên tập test
y_pred = svc_model.predict(X_test_scaled)

# Đánh giá độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Hiển thị báo cáo phân loại
print(classification_report(y_test, y_pred))

# Hiển thị ma trận nhầm lẫn
ConfusionMatrixDisplay.from_estimator(svc_model, X_test_scaled, y_test)
plt.show()

# Tạo lưới điểm để vẽ
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-5, X[:, 0].max()+5, 100),
                     np.linspace(X[:, 1].min()-5, X[:, 1].max()+5, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)

# Dự đoán cho từng điểm trên lưới
Z = svc_model.predict(grid_scaled).reshape(xx.shape)

# Vẽ vùng phân loại
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Vẽ dữ liệu thực tế
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('SVM Classification - Gender Prediction')
plt.legend(handles=scatter.legend_elements()[0], labels=['Female', 'Male'])
plt.show()
