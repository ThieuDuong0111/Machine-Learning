import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Dữ liệu (Kích thước, Trọng lượng, Loại)
data = {
    "Size": [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5],
    "Weight": [150, 160, 170, 180, 190, 200, 210, 220],
    "Fruit": ["Apple", "Apple", "Apple", "Apple", "Orange", "Orange", "Orange", "Orange"]
}

df = pd.DataFrame(data)

# Chuyển đổi nhãn thành số (Apple = 0, Orange = 1)
df["Fruit"] = df["Fruit"].map({"Apple": 0, "Orange": 1})

# Tạo tập dữ liệu đầu vào (X) và nhãn (y)
X = df[["Size", "Weight"]]
y = df["Fruit"]

# Tạo mô hình Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
clf.fit(X, y)

plt.figure(figsize=(10, 6))
tree.plot_tree(clf, feature_names=["Size", "Weight"], class_names=["Apple", "Orange"], filled=True)
plt.show()

fruit_prediction = clf.predict([[9, 185]])
if fruit_prediction[0] == 0:
    print("Dự đoán: 🍏 Apple")
else:
    print("Dự đoán: 🍊 Orange")

from sklearn.metrics import accuracy_score

# Dự đoán trên dữ liệu huấn luyện
y_pred = clf.predict(X)

# Tính độ chính xác
accuracy = accuracy_score(y, y_pred)
print(f"Độ chính xác của mô hình: {accuracy:.2f}")
