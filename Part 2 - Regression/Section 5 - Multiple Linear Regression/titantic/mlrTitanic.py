# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 21:27:45 2019

@author: gillespie.evan@gmail.com

Multiple Linear Regression Titantic flavor!

Caveats - data must be prepared...

Dummy Variable Trap...

Always omit one dummy variable...

p-value 0.05 is the most commonly used

All-in
Stepwise regression
    Backward Selection *fastest
    Forward Selection
    Bi-Directional Selection
Score Comparison
All Possible Models

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('train.csv')

# independant variables upper bound not inclusive
X = dataset.iloc[:, 2:12].values

# dependant variables
y = dataset.iloc[:, 1].values

# 1 - survived, 2 - pclass, 3 - name, 4 - sex, 5 - Age,
# 6 - Sibs, 7 - Parents, 8 - Ticket, 9 - Fare, 10 - Cabin,
# 11 - Port 

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 2] = labelencoder.fit_transform(X[:, 12])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 0:1])
X[:, 0:1] = imputer.transform(X[:, 0:1])

# split into test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# FEATURE SCALING
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""


