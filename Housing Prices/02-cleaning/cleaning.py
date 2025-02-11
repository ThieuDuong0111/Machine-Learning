''' 
Overview
+ remove skewenes of target feature
+ remove skewenes of numeric features is exists
+ handle missing values in categorical features
+ handle missing values in numerical features
+ feature selection 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# from __future__ import division

sns.set_style("whitegrid")

train = pd.read_csv("Housing Prices/data/train.csv")
test = pd.read_csv("Housing Prices/data/test.csv")

'''Target variable'''
target = train['SalePrice']
target_log = np.log1p(train['SalePrice'])

'''Concat train and test dataset in order for pre-processing'''
'''In order to apply transformations on data, we have to concatenate both datasets: train and test'''
# drop target variable from train dataset
train = train.drop(["SalePrice"], axis=1)
data = pd.concat([train, test], ignore_index=True)

'''Split dataframe into numeric and categorical'''
# save all categorical columns in list
categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']

# dataframe with categorical features
data_cat = data[categorical_columns]
# dataframe with numerical features
data_num = data.drop(categorical_columns, axis=1)

print(data_num.head(1))
print(data_num.describe())
print(data_cat.head(1))

'''Reduce skewness for numeric features'''
# Tính Skewness của từng cột số trong data_num
data_num.dropna
data_num_skew = data_num.apply(lambda x: skew(x.dropna()))
data_num_skew = data_num_skew[data_num_skew > .75]

# Áp dụng log1p để giảm Skewness
data_num[data_num_skew.index] = np.log1p(data_num[data_num_skew.index])

print(data_num_skew)

'''Missing values'''
# handling missing values in numerical columns
data_num.drop

data_len = data_num.shape[0]

# check what is percentage of missing values in categorical dataframe
for col in data_num.columns.values:
    missing_values = data_num[col].isnull().sum()
    #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 
    
    # drop column if there is more than 50 missing values
    if missing_values > 50:
        #print("droping column: {}".format(col))
        data_num = data_num.drop(col, axis = 1)
    # if there is less than 50 missing values than fill in with median valu of column
    else:
        #print("filling missing values with median in column: {}".format(col))
        data_num = data_num.fillna(data_num[col].median())

# handling missing values in categorical columns
data_len = data_cat.shape[0]

# check what is percentage of missing values in categorical dataframe
for col in data_cat.columns.values:
    missing_values = data_cat[col].isnull().sum()
    #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 
    
    # drop column if there is more than 50 missing values
    if missing_values > 50:
        print("droping column: {}".format(col))
        data_cat.drop(col, axis = 1)
    # if there is less than 50 missing values than fill in with median valu of column
    else:
        #print("filling missing values with XXX: {}".format(col))
        #data_cat = data_cat.fillna('XXX')
        pass

print(data_cat.describe())
print(data_num.describe())
data_cat_dummies = pd.get_dummies(data_cat)

'''Merge and save for further use'''
print(data_num.shape)
print(data_cat.shape)

data = pd.concat([data_num, data_cat_dummies], axis=1)

train = data.iloc[:len(train)-1]
train = train.join(target_log)

test = data.iloc[len(train)+1:]

train.to_pickle("Housing Prices/data/train.pkl")
test.to_pickle("Housing Prices/data/test.pkl")

print(data.columns.values)