#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:31:53 2017

@author: zkidkid
"""

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

#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X = sc_X.fit_transform(X)
#sc_y = StandardScaler()
#y = sc_y.fit_transform(y)

# Fitting Random Forest Regression to dataset
from sklearn.ensemble import RandomForestRegressor
regressor =  RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)


predict_salary = regressor.predict(6.5)
print("Salary for level 6.5 = ",predict_salary)

#Visualize


X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (Decision Tree)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()