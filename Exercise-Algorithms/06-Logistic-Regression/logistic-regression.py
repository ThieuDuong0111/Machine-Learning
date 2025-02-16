import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1️⃣ Tạo dữ liệu giả lập (sử dụng bộ dữ liệu giả lập để dự đoán liệu một sinh viên có đậu đại học hay không dựa vào số giờ học)
data = {
    "Hours_Studied": [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
    "Pass": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# 2️⃣ Tách dữ liệu thành X (feature) và y (label)
X = df[["Hours_Studied"]]
y = df["Pass"]

# 3️⃣ Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Huấn luyện mô hình Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 5️⃣ Dự đoán và đánh giá
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# 6️⃣ Vẽ đồ thị Sigmoid
x_range = np.linspace(1, 7, 100).reshape(-1, 1)  # Giá trị từ 1 đến 7
y_prob = log_reg.predict_proba(x_range)[:, 1]  # Xác suất đậu đại học

plt.scatter(df["Hours_Studied"], df["Pass"], color="red", label="Actual Data")
plt.plot(x_range, y_prob, color="blue", label="Sigmoid Curve")

plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression for Exam Prediction")
plt.legend()
plt.show()
