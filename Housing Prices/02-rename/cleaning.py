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

from __future__ import division

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