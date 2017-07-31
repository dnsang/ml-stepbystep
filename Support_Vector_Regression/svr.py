#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:41:57 2017

@author: zkidkid
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:,2].values

#Feature Scaling ( SVR model doesn't built-in feature scaling)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Fitting SVR to dataset
from sklearn.svm import SVR
regressor =  SVR(kernel = 'rbf')
regressor.fit(X,y)


predict_salary = regressor.predict(sc_X.transform(np.array([[6.5]])))
print("Salary for level 6.5 = ",sc_y.inverse_transform(predict_salary))

#Visualize
plt.scatter(X,y,color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()