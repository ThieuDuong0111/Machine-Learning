# import library
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# import data
data = pd.read_csv("Exercise/Linear-Regression/linear-regression-dataset.csv")
print(data.info())
print(data.head())
#print(data.describe())

plt.scatter(data.deneyim,data.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
# plt.show()

# linear regression model
linear_reg = LinearRegression()

x = data.deneyim.values.reshape(-1,1)
y = data.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

print('R sq: ', linear_reg.score(x, y))
print('Correlation: ', math.sqrt(linear_reg.score(x, y)))

# prediction

print("Coefficient for X: ", linear_reg.coef_)
print("Intercept for X: ", linear_reg.intercept_)
print("Regression line is: y = " + str(linear_reg.intercept_[0]) + " + (x * " + str(linear_reg.coef_[0][0]) + ")")

array = np.array([11, 0]).reshape(-1,1)
print(linear_reg.predict(array))

# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # deneyim

plt.scatter(x,y)
y_head = linear_reg.predict(array)  # maas
plt.plot(array, y_head,color = "red")
array = np.array([100]).reshape(-1,1)
linear_reg.predict(array)
plt.show()

y_head = linear_reg.predict(x)  # maas
print("r_square score: ", r2_score(y,y_head))
