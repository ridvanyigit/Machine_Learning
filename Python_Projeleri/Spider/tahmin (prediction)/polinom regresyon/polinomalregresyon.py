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


# Polynominal Regression
poly_reg = PolynomialFeatures(degree= 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


# Scatter Plot görselleştirme (Polinom Regresyon)
plt.scatter(X,Y, color= 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color= 'blue')
plt.title('Polinom Regresyon')
plt.xlabel('Pozisyon Seviyesi')
plt.ylabel('Maaş')
plt.show()

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))





