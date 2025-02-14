import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Tạo dữ liệu giả lập: Số năm kinh nghiệm, Tuổi, Trình độ học vấn và Lương
data = pd.DataFrame({
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [22, 23, 25, 27, 29, 30, 32, 35, 37, 40],
    'Education': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],  # 1: High School, 2: Associate, 3: Bachelor, 4: Master, 5: PhD
    'Salary': [30, 35, 40, 45, 50, 60, 65, 70, 75, 80]
})

# Chia thành feature X và target y
X = data[['Experience', 'Age', 'Education']].values
y = data['Salary'].values.reshape(-1,1)

# Chuẩn hóa dữ liệu (Vì SVR bị ảnh hưởng bởi scale)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)

# Huấn luyện mô hình SVR với kernel 'rbf'
svr_model = SVR(kernel='rbf')
svr_model.fit(X_scaled, y_scaled.ravel())  # ravel() để chuyển y thành vector 1D

# Tạo tập dữ liệu test
X_test = np.array([[5, 28, 3], [7, 33, 4]])  # 5 năm KN, 28 tuổi, Bachelor | 7 năm KN, 33 tuổi, Master
X_test_scaled = sc_X.transform(X_test)

# Dự đoán lương với SVR
y_pred_scaled = svr_model.predict(X_test_scaled)

# Chuyển kết quả về scale ban đầu
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1,1))

# In kết quả
for i, salary in enumerate(y_pred):
    print(f"Predicted Salary for {X_test[i]}: ${salary[0]:.2f}")

# Vẽ đồ thị (Chỉ vẽ với Experience vì dữ liệu có nhiều chiều)
plt.scatter(data['Experience'], data['Salary'], color='red', label='Dữ liệu thực tế')
plt.scatter(X_test[:,0], y_pred, color='blue', marker='x', s=100, label='Dự đoán của SVR')
plt.xlabel('Experience (Năm)')
plt.ylabel('Salary (Lương)')
plt.title('Support Vector Regression (SVR) với nhiều features')
plt.legend()
plt.show()
