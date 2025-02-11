import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("Exercise/04-Decision-Tree-Regression/decision-tree-regression-dataset.csv", header=None)
print(data.info())
print(data.head())
#print(data.describe())

x = data.iloc[:,[0]].values.reshape(-1,1) # lấy cột 0
y = data.iloc[:,[1]].values.reshape(-1,1) # lấy cột 1

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

print(tree_reg.predict(np.array([5.5]).reshape(-1,1)))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
#print(x)
y_head = tree_reg.predict(x_)
#print(y_head)

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()

y_head = tree_reg.predict(x)

print("r_square score: ", r2_score(y,y_head))