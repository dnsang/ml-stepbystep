#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:53:44 2017

@author: zkidkid
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:,1].values


# Spliting the dataset into the training set & test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

#Fitting slr to training set
from sklearn.linear_model import LinearRegression 

regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_predict = regressor.predict(X_test)
#year_exp = np.matrix([[0],[15]])
#y_predict_outbound=regressor.predict(year_exp)
#
#print(year_exp,y_predict_outbound)

#Visualize Training Set Result
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#Visualize test set result

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()
