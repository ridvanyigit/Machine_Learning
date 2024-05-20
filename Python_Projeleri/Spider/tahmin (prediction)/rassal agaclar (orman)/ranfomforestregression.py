#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:17:23 2024

@author: ridvanyigit
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Veri Yükleme
veriler = pd.read_csv('maaslar.csv')

#Slicing (DataFrame dilimleme)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPy Array (dizi) dönüstürme
X = x.values
Y = y.values



from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators= 10, random_state= 0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X, Y, color= 'red')
plt.plot(X, rf_reg.predict(X), color= 'blue')



