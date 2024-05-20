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




#support vektor regression islemlerinde mutlaka ölceklendirme (Scaling) yapilmalidir
from sklearn.preprocessing import StandardScaler


sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))
                                                                               # y = a + f(x).b + e  

from sklearn.svm import SVR

svr_reg = SVR(kernel= 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli,y_olcekli, color= 'red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli), color= 'blue')
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))





