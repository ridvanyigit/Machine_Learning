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
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('maaslar_yeni.csv')
print(df.head(10))

X = df[['UnvanSeviyesi', 'Kidem', 'Puan']]
y = df['maas']

numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.corr())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)






# Multiple Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)                #--> Zaten OLS Raporu olusturulurken, ayrica r2 ve hata kareler ortalamasi hesaplanmasi gerekmez

X_train_sm = sm.add_constant(X_train)              #--> ifadesi bağımsız değişkenlerin(X) bir sabit terimle(1) birleştirilmesini sağlar.
model_lr = sm.OLS(y_train, X_train_sm).fit()       #--> OLS (Ordinary Least Squares - En Küçük Kareler) Yöntemi
print(model_lr.summary())






# Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_train)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_poly, y_train)

X_test_poly = poly_features.transform(X_test)
y_pred_poly = poly_reg_model.predict(X_test_poly)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)            #--> Zaten OLS Raporu olusturulurken, ayrica r2 ve hata kareler ortalamasi hesaplanmasi gerekmez

X_poly_sm = poly_features.fit_transform(X_train)
model_poly = sm.OLS(y_train, X_poly_sm).fit()      #--> OLS Raporu dogursal regresyonlara uygulanir.
print(model_poly.summary())







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






# Görselleştirme
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.title("Gerçek vs Tahmin Edilen (Linear Regression)")
sns.scatterplot(x=y_test, y=y_pred_lr)
plt.xlabel('Gerçek')
plt.ylabel('Tahmin Edilen')

plt.subplot(2, 3, 2)
plt.title("Gerçek vs Tahmin Edilen (Polynomial Regression)")
sns.scatterplot(x=y_test, y=y_pred_poly)
plt.xlabel('Gerçek')
plt.ylabel('Tahmin Edilen')

plt.subplot(2, 3, 3)
plt.title("Gerçek vs Tahmin Edilen (SVR)")
sns.scatterplot(x=y_test, y=y_pred_svr)
plt.xlabel('Gerçek')
plt.ylabel('Tahmin Edilen')

plt.subplot(2, 3, 4)
plt.title("Gerçek vs Tahmin Edilen (Decision Tree)")
sns.scatterplot(x=y_test, y=y_pred_dt)
plt.xlabel('Gerçek')
plt.ylabel('Tahmin Edilen')

plt.subplot(2, 3, 5)
plt.title("Gerçek vs Tahmin Edilen (Random Forest)")
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel('Gerçek')
plt.ylabel('Tahmin Edilen')

plt.tight_layout()
plt.show()


