import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Exercise/02-Multiple-Linear-Regression/multiple-linear-regression-dataset.csv")
print(data.info())
print(data.head())
#print(data.describe())

x = data.iloc[:,[0,2]].values  # Chọn cột thứ 0 (deneyim - kinh nghiệm) và cột thứ 2 (yaş - tuổi) làm features
y = data.maas.values.reshape(-1,1)  # Biến phụ thuộc là mức lương (maas), reshape thành mảng 2D

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0:",multiple_linear_regression.intercept_)
print("b1 and b2: ", multiple_linear_regression.coef_)


#predict
x_ = np.array([[10,35],[5,35]])
print(multiple_linear_regression.predict(x_))

y_head = multiple_linear_regression.predict(x) 

print("r_square score: ", r2_score(y,y_head))

# Lấy hệ số hồi quy (b1, b2, ...)
b0 = multiple_linear_regression.intercept_[0]  # Hệ số chặn
b1, b2 = multiple_linear_regression.coef_[0]  # Hệ số của các feature

# In phương trình
print(f"Phương trình hồi quy: y = {b0:.2f} + ({b1:.2f} * Experience) + ({b2:.2f} * Age)")
print(b0 + b1 * 10 + b2 * 35)

# Dự đoán trên tập dữ liệu gốc
y_pred = multiple_linear_regression.predict(x)

# Đánh giá mô hình
r2 = r2_score(y, y_pred)
print(f"R-squared Score: {r2:.2f}")

# Vẽ đồ thị 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Vẽ điểm dữ liệu thực tế
ax.scatter(x[:, 0], x[:, 1], y, color='red', label='Actual Data')

# Vẽ mặt phẳng hồi quy
X1_range = np.linspace(min(x[:, 0]), max(x[:, 0]), 10)
X2_range = np.linspace(min(x[:, 1]), max(x[:, 1]), 10)
X1_mesh, X2_mesh = np.meshgrid(X1_range, X2_range)
Y_pred_mesh = b0 + b1 * X1_mesh + b2 * X2_mesh
ax.plot_surface(X1_mesh, X2_mesh, Y_pred_mesh, alpha=0.5, color='blue')

# Thiết lập trục
ax.set_xlabel("Experience (Years)")
ax.set_ylabel("Age (Years)")
ax.set_zlabel("Salary")
ax.set_title("Multiple Linear Regression")
ax.legend()

plt.show()
