import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
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

'''Ví dụ của Chat-GPT'''
# Tạo DataFrame
data = {
    'Kinh nghiệm': [1, 3, 5, 7, 9, 11],
    'Tuổi': [22, 25, 28, 32, 35, 40],
    'Mức lương': [1000, 3000, 5000, 7000, 9000, 11000]
}
df = pd.DataFrame(data)

# Tách features và target
X = df[['Kinh nghiệm', 'Tuổi']]
y = df['Mức lương']

# Train Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=3)
model.fit(X, y)

# Predict
print(model.predict(np.array([[11,35], [10,30]])))

# Vẽ cây quyết định
plt.figure(figsize=(10,5))
plot_tree(model, feature_names=['Kinh nghiệm', 'Tuổi'], filled=True)
plt.show()