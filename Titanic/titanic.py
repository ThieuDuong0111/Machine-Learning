import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv(r"D:\Projects\Python\Machine-Learning\Titanic\train.csv")
print(train_data.head())

test_data = pd.read_csv(r"D:\Projects\Python\Machine-Learning\Titanic\test.csv")
print(test_data.head())

#loc: lọc ra những hàng có giá trị là 'female'
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

#loc: lọc ra những hàng có giá trị là 'male'
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

# Sử dụng thuật toán random forest model
# Lấy cột "Survived" làm biến mục tiêu (y), chứa nhãn sống sót (1) hoặc không (0)
y = train_data["Survived"]

# Chọn các đặc trưng (features) sẽ sử dụng để huấn luyện mô hình
features = ["Pclass", "Sex", "SibSp", "Parch"]

# Biến đổi dữ liệu dạng chuỗi (Sex) thành dạng số bằng phương pháp One-Hot Encoding
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Khởi tạo mô hình Random Forest với 100 cây quyết định, độ sâu tối đa là 5
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Huấn luyện mô hình với dữ liệu huấn luyện (X) và nhãn (y)
model.fit(X, y)

# Dự đoán kết quả trên tập kiểm tra (X_test)
predictions = model.predict(X_test)

# Tạo DataFrame chứa ID hành khách và kết quả dự đoán
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# Lưu kết quả dự đoán vào file CSV để nộp
output.to_csv('Titanic/submission.csv', index=False)
print("Your submission was successfully saved!")