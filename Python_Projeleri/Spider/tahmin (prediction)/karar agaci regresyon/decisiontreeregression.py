#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 22:19:17 2024

@author: ridvanyigit
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Veri Yükleme
veriler = pd.read_csv('maaslar.csv')

#Slicing (DataFrame dilimleme)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPy Array (dizi) dönüstürme
X = x.values
Y = y.values



#Decision Tree Regression (Karar Agaci Regresyon)
r_dt = DecisionTreeRegressor(random_state= 0)
r_dt.fit(X,Y)

Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y)
plt.plot(x, r_dt.predict(X), color= 'blue')

plt.plot(x,r_dt.predict(Z), color= 'green')
plt.plot(x,r_dt.predict(K), color= 'yellow')


print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))


























