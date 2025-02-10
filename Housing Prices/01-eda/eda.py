import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno # Một thư viện trong Python giúp trực quan hóa dữ liệu bị thiếu trong DataFrame bằng cách vẽ biểu đồ

train = pd.read_csv("Housing Prices/data/train.csv")

# Show all columns of train data
print(train.columns)

# Count columns
print(len(train.columns))

# Show first 5 rows 
print(train.head())

# Show last 5 rows
print(train.tail())

# Show data type of per column
print(train.dtypes)

# Describe specific column
print(train.SaleType.describe())
print(train.SalePrice.describe())

sns.set_style("whitegrid")

plt.figure(figsize=(10,5))

# Divide the drawing area into 1 row, 2 columns, and select the first cell to draw.
plt.subplot(1,2,1)

# histplot: Draw histogram
# bins=50: Divide the interval into 50 columns
# ked=True: Show additional density lines (Kernel Density Estimation - KDE) to represent data distribution.
sns.histplot(train.SalePrice, bins=50, kde=True)
plt.title('Original')

# Divide the drawing area into 1 row, 2 columns, and select the second cell to draw.
plt.subplot(1,2,2)

# histplot: Draw histogram
# bins=50: Divide the interval into 50 columns
# ked=True: Show additional density lines (Kernel Density Estimation - KDE) to represent data distribution.
sns.histplot(np.log1p(train.SalePrice), bins=50, kde=True)
plt.title('Log transformed')

plt.tight_layout()

# skew: Tells whether the data distribution is symmetrical or not
print(train.SalePrice.skew())

# kurt: Helps measure the kurtosis of the data distribution
print(train.SalePrice.kurt())

print(train["GrLivArea"])
var = 'GrLivArea'
data = pd.concat([train["SalePrice"], train[var]], axis=1)
print(data.head())

# ylim=(0,800000): Y-axis limit from 0 to 800,000 (makes it easier to see)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# train.corr() tính toán ma trận tương quan (correlation matrix) giữa các biến số trong tập dữ liệu train.
# Tương quan đo lường mối quan hệ giữa hai biến:
# 1.0: Tương quan hoàn toàn dương (biến này tăng, biến kia tăng).
# -1.0: Tương quan hoàn toàn âm (biến này tăng, biến kia giảm).
# 0.0: Không có tương quan.
corr_matrix = train.corr(numeric_only=True)
sns.set_theme(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_matrix, vmax=.8, square=True)

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set_theme(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

# sns.set_theme()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# sns.pairplot() để vẽ biểu đồ cặp (pair plot) giữa các biến số trong danh sách cols, giúp trực quan hóa mối quan hệ giữa chúng
sns.pairplot(train[cols], height = 2.5)

# train.isnull() → Trả về một DataFrame với giá trị True nếu bị thiếu (NaN), False nếu có giá trị.
# sum() → Đếm số lượng giá trị NaN trong từng cột.
# sort_values(ascending=False) → Sắp xếp danh sách theo thứ tự giảm dần, tức là cột nào có nhiều giá trị thiếu nhất sẽ được xếp đầu tiên.
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(10))

msno.matrix(train.sample(500))
msno.bar(train.sample(500))
msno.heatmap(train)

print(train.shape)
# pd.get_dummies() là một phương pháp của pandas giúp chuyển đổi các biến phân loại (categorical features) thành biến số (numerical features).
# Nó mã hóa One-Hot Encoding, tức là với mỗi giá trị trong một cột phân loại, nó tạo ra một cột mới có giá trị 0 hoặc 1.
# Mục đích: Giúp Machine Learning có thể sử dụng dữ liệu dạng số thay vì chữ.
train = pd.get_dummies(train)
print(train['SalePrice'])

# plt.show()