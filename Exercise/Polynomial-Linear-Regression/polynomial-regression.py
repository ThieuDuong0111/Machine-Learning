# import library
# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# import data
data = pd.read_csv("Exercise/Polynomial-Linear-Regression/polynomial-regression-dataset.csv")
print(data.info())
print(data.head())
#print(data.describe())

x = data.araba_fiyat.values.reshape(-1,1)
y = data.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba_max_hiz")
plt.ylabel("araba_fiyat")
# plt.show()

polynominal_regression = PolynomialFeatures(degree=4)
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