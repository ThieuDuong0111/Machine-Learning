import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

data = pd.read_csv("Exercise/05-Random-Forest-Regression/random-forest-regression-dataset.csv", header=None)
print(data.info())
print(data.head())
#print(data.describe())

x = data.iloc[:,[0]].values.reshape(-1,1) # lấy cột 0
y = data.iloc[:,[1]].values.reshape(-1,1) # lấy cột 1

rf = RandomForestRegressor(n_estimators = 100, random_state= 42) 
rf.fit(x,y)

print("7.8 seviyesinde fiyatın ne kadar olduğu: ",rf.predict(np.array([7.8]).reshape(-1,1)))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

# visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()

y_head = rf.predict(x)
from sklearn.metrics import r2_score
print("r_score: ", r2_score(y,y_head))