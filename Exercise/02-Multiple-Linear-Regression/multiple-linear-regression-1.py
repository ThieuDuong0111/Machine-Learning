import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

# Tạo dữ liệu mẫu (hoặc đọc từ CSV)
data = pd.DataFrame({
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [22, 25, 28, 30, 32, 35, 38, 40, 42, 45],
    'Education': [12, 14, 16, 12, 14, 16, 18, 12, 14, 16],  # Số năm học
    'Salary': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]  # Lương
})

# Chọn features và target
X = data[['Experience', 'Age', 'Education']].values  # 3 features
y = data['Salary'].values.reshape(-1,1)

# Train mô hình
model = LinearRegression()
model.fit(X, y)

# Lấy hệ số hồi quy
b0 = model.intercept_[0]
b1, b2, b3 = model.coef_[0]
print(f"Phương trình hồi quy: y = {b0:.2f} + ({b1:.2f} * Experience) + ({b2:.2f} * Age) + ({b3:.2f} * Education)")

# Vẽ biểu đồ 3D (cố định Education = 14)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Lọc dữ liệu với Education = 14
subset = data[data['Education'] == 14]

# Vẽ điểm dữ liệu gốc
ax.scatter(subset['Experience'], subset['Age'], subset['Salary'], c='red', label='Dữ liệu thực tế')

# Tạo lưới điểm để vẽ mặt phẳng hồi quy
exp_range = np.linspace(data['Experience'].min(), data['Experience'].max(), 10)
age_range = np.linspace(data['Age'].min(), data['Age'].max(), 10)
exp_grid, age_grid = np.meshgrid(exp_range, age_range)
edu_fixed = 14  # Giá trị cố định cho Education
salary_pred = b0 + b1*exp_grid + b2*age_grid + b3*edu_fixed

# Vẽ mặt phẳng hồi quy
ax.plot_surface(exp_grid, age_grid, salary_pred, alpha=0.5, cmap='coolwarm')

ax.set_xlabel('Experience')
ax.set_ylabel('Age')
ax.set_zlabel('Salary')
ax.set_title('Multiple Linear Regression (Fixed Education = 14)')
plt.legend()
plt.show()
