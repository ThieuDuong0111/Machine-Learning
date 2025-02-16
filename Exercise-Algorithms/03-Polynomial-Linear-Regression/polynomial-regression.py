# import library
# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# import data
data = pd.read_csv("Exercise-Algorithms/03-Polynomial-Linear-Regression/polynomial-regression-dataset.csv")
print(data.info())
print(data.head())
#print(data.describe())

x = data.araba_fiyat.values.reshape(-1,1)
y = data.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba_max_hiz")
plt.ylabel("araba_fiyat")
# plt.show()

'''
Tạo bộ biến đổi dữ liệu thành đa thức
degree=4 nghĩa là mô hình sẽ sử dụng đa thức bậc 4 (𝑥,𝑥2,𝑥3,𝑥4).
Ban đầu, dữ liệu X chỉ có dạng:
𝑋=[𝑥]
Sau khi dùng PolynomialFeatures(degree=4), dữ liệu sẽ biến đổi thành:
𝑋=[1,𝑥,𝑥2,𝑥3,𝑥4]
(Thêm các bậc cao hơn để mô hình phi tuyến tính).
'''
polynominal_regression = PolynomialFeatures(degree=4)

# fit_transform(x, y): Áp dụng biến đổi đa thức lên x.
x_polynomial = polynominal_regression.fit_transform(x,y)

linear_regression = LinearRegression()
linear_regression.fit(x_polynomial,y)

y_head2 = linear_regression.predict(x_polynomial)

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.scatter(x,y)
plt.xlabel("araba_max_hiz")
plt.ylabel("araba_fiyat")
plt.show()

print("r_square score: ", r2_score(y,y_head2))