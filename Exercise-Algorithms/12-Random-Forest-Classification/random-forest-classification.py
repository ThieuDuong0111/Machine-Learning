import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Tạo dữ liệu
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

# Tạo mô hình Random Forest Classifier
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X, y)

fruit_prediction = clf.predict([[9, 185]])
if fruit_prediction[0] == 0:
    print("Dự đoán: 🍏 Apple")
else:
    print("Dự đoán: 🍊 Orange")

# Dự đoán trên dữ liệu huấn luyện
y_pred = clf.predict(X)

# Tính độ chính xác
accuracy = accuracy_score(y, y_pred)
print(f"Độ chính xác của mô hình: {accuracy:.2f}")

# Lấy ra mức độ quan trọng của các đặc trưng
importances = clf.feature_importances_

# Hiển thị mức độ quan trọng
for feature, importance in zip(["Size", "Weight"], importances):
    print(f"{feature}: {importance:.2f}")
