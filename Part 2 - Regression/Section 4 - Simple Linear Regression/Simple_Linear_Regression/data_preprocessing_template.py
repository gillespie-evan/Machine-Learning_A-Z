# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:17:50 2019
@author: gillespie.evan@gmail.com

Simple Linear Regression Salary Data

Simple Linear Regression y = b0 + b1*x1 #nums are subscript

y is the dependent variable (DV) what are we looking for?
x1 is the dependent variable (IV) our data to make predictions

b1 is the coefficient that connects x1 and y
b0 is the Constant

The constant is where the DV starts on the X axis
b1 is the slope

Ordinary Least Squares

SUM (y - y^2)^2 -> min

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')

# independant variables upper bound not inclusive
# x is a matrix of IVs
X = dataset.iloc[:, :-1].values

# dependant variables
# y is the vector
y = dataset.iloc[:, 1].values

# split into test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# FEATURE SCALING if the library doesn't take care of it
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

 # Fitting SLR to the training set
 from sklearn.linear_model import LinearRegression
 regressor = LinearRegression()
 
 regressor.fit(X_train, y_train)

















