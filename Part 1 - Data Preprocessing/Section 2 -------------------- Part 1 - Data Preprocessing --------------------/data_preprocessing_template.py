# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 12:17:50 2019

@author: gillespie.evan@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Data.csv')

# independant variables
X = dataset.iloc[:, :-1].values

# dependant variables
y = dataset.iloc[:, 3].values

# resolve missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

# dummy encoding to avoid bias IE Germany > Spain
# We want to avoid the model into thinking that non-numerical features have
# greater values than other features

oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# split into test and train data
from sklearn.model_selection import train_test_split

# 20%-30% for test size, not using random_state = 0 because I want diff results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# FEATURE SCALING
#
# Need to resolve Euclidean distance with different scales get it to -1 to +1
# two common ways Standardisation and Normalistation
# Xstand = x - mean(x)/standard deviation(x)
# Xnorm = x - min(x)/max(x) - min(x) - used this for climate sensors...
# We don't need to scale y because it's 1/0 Yes/No
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




















