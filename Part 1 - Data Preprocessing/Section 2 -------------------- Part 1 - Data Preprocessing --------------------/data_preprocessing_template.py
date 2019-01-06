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

#independantvariables
X = dataset.iloc[:, :-1].values

#dependant variables
y = dataset.iloc[:, 3].values

#resolve missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
