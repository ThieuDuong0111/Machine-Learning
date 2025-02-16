import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import graphviz

from sklearn.metrics import r2_score

data = pd.read_csv("Exercise-Algorithms/05-Random-Forest-Regression/random-forest-regression-dataset.csv", header=None)
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

'''Ví dụ của Chat-GPT'''
# 📌 Tạo DataFrame
data = {
    'Kinh nghiệm': [1, 3, 5, 7, 9, 11],
    'Tuổi': [22, 25, 28, 32, 35, 40],
    'Mức lương': [1000, 3000, 5000, 7000, 9000, 11000]
}
df = pd.DataFrame(data)

# 📌 Tách features và target
X = df[['Kinh nghiệm', 'Tuổi']]
y = df['Mức lương']

# 📌 Khởi tạo Random Forest Regressor với 10 cây quyết định
rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
rf_regressor.fit(X, y)

# 📌 Dự đoán mức lương cho một người có 6 năm kinh nghiệm và 30 tuổi
prediction = rf_regressor.predict([[6, 30]])
print("Dự đoán mức lương cho 6 năm kinh nghiệm và 30 tuổi:", prediction[0])

# 📌 Dự đoán cho toàn bộ dữ liệu và vẽ biểu đồ
y_pred = rf_regressor.predict(X)

plt.scatter(X['Kinh nghiệm'], y, color="red", label="Dữ liệu thật")
plt.scatter(X['Kinh nghiệm'], y_pred, color="blue", label="Dự đoán")
plt.xlabel("Kinh nghiệm (năm)")
plt.ylabel("Mức lương ($)")
plt.legend()
plt.show()

# # 📌 Chọn một cây trong rừng để vẽ (ví dụ: cây số 0)
# tree = rf_regressor.estimators_[0]

# # 📌 Xuất cây dưới dạng DOT format
# dot_data = export_graphviz(
#     tree, out_file=None, 
#     feature_names=['Kinh nghiệm', 'Tuổi'],  
#     filled=True, rounded=True,
#     special_characters=True
# )

# # 📌 Vẽ Decision Tree
# graph = graphviz.Source(dot_data)
# graph.render("Exercise-Algorithms/05-Random-Forest-Regression/decision_tree.pdf")  # Lưu cây quyết định thành file PDF
# graph.view()  # Hiển thị cây quyết định