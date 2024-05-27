#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:56:54 2024

@author: ridvanyigit
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Hata Kareler Ortalaması = MSE

df = pd.read_csv('maaslar_yeni.csv')
print(df.head(10))

X = df[['UnvanSeviyesi', 'Kidem', 'Puan']]
y = df['maas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_train)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_poly, y_train)

X_test_poly = poly_features.transform(X_test)
y_pred_poly = poly_reg_model.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Support Vektor Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel='rbf')
svr.fit(X_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Decision Tree Regression
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Sonuclarin Karsilastirilmasi
results = pd.DataFrame({
    'Model': [
        'Linear Regression', 
        'Polynomial Regression', 
        'Support Vector Regression', 
        'Decision Tree Regression', 
        'Random Forest Regression'
        ],
    
    'MSE': [
        mse_lr, 
        mse_poly, 
        mse_svr, 
        mse_dt, 
        mse_rf
        ],
    
    'R2 Score': [
        r2_lr, 
        r2_poly, 
        r2_svr, 
        r2_dt, 
        r2_rf]
})

print(results)























