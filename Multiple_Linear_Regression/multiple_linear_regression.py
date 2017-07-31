#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:46:58 2017

@author: zkidkid
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy var trap
X = X[:,1:]

# Spliting the dataset into the training set & test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

#Fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train ,y_train)

#Predicting 
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X_opt = np.append(arr = np.ones((50,1)), values = X ,axis=1)
X_opt_train,X_opt_test,y_opt_train,y_opt_test = train_test_split(X,y, test_size = 0.2, random_state=0)

data = X_opt_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y_opt_train, exog=data).fit()
regressor_OLS.summary()

data = X_opt_train[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y_opt_train, exog=data).fit()
regressor_OLS.summary()

data = X_opt_train[:,[0,1,4,5]]
regressor_OLS = sm.OLS(endog=y_opt_train, exog=data).fit()
regressor_OLS.summary()

data = X_opt_train[:,[0,1,4]]
regressor_OLS = sm.OLS(endog=y_opt_train, exog=data).fit()
regressor_OLS.summary()


y_pred_opt = regressor_OLS.predict(X_opt_test[:,[0,1,4]])

