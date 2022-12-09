# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:42:092020

@author: SAMPCAC
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("deneme.csv",sep=";")
x = data.iloc[:,[1,2]]
y = data.iloc[:,[3]]

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.15,random_state=32)
mlr = LinearRegression()

mlr.fit(x_train,y_train)
pred=mlr.predict(x_test)
print("Tahmin : ",pred)
print("\n")
print("b0:",mlr.intercept_)
print("b1,b2:",mlr.coef_)
scor=mlr.score(x_test,y_test)
print("R^2 : ",scor)

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

model1=sm.OLS(endog=x,exog=y)

result=model1.fit()

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,pred)
print("mse : ",mse)
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, pred))
print("rmse : ",rmse)




































