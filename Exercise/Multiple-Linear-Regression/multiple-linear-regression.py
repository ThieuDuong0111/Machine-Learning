import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Exercise/Multiple-Linear-Regression/multiple-linear-regression-dataset.csv")
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