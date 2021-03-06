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

# independant variables upper bound not inclusive
X = dataset.iloc[:, :-1].values

# dependant variables
y = dataset.iloc[:, 3].values

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


















