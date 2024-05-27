import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Verilerin yüklenmesi
veriler = pd.read_csv('odev_tenis.csv')
print('Veriler:\n', veriler)

# Encoder: Kategorik -> Numeric
veriler_num = veriler.apply(LabelEncoder().fit_transform)

c = veriler_num.iloc[:, :1]
ohe = OneHotEncoder()
c = ohe.fit_transform(c).toarray()

havadurumu = pd.DataFrame(data=c, index=range(14), columns=['o', 'r', 's'])
sonveriler = pd.concat([havadurumu, veriler.iloc[:, 1:3]], axis=1)
sonveriler = pd.concat([sonveriler, veriler_num.iloc[:, -2:]], axis=1)
print('\nSon Veriler: \n', sonveriler)

# Verilerin eğitim ve test için bölünmesi
x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:, :-1], sonveriler.iloc[:, -1:], test_size=0.33, random_state=0)

# Çoklu doğrusal regresyon uygulama
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print('Birinci tahmin sonucu: ', y_pred)

# Modelin başarısının değerlendirilmesi (Olasılık değeri (p-değeri) ölçümü)
# Backward Elimination (Geriye Doğru Eleme)
X = np.append(arr=np.ones((14, 1)).astype(int), values=sonveriler.iloc[:, :-1], axis=1)
X_l = sonveriler.iloc[:, :-1].values
y = sonveriler.iloc[:, -1].values

r_ols = sm.OLS(endog=y, exog=X_l).fit()
print('\nBirinci Değerlendirme Raporu: \n', r_ols.summary())

# En yüksek p-değerine sahip olan 0. indeksteki sütunu kaldırma
X_l = sonveriler.iloc[:, 1:-1].values
r_ols = sm.OLS(endog=y, exog=X_l).fit()
print('\nİkinci Değerlendirme Raporu: \n', r_ols.summary())

# Backward Elimination adımlarına devam edilerek modelin iyileştirilmesi sağlanabilir

# Çoklu doğrusal regresyon uygulama (güncellenmiş model)
x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:, 1:-1], sonveriler.iloc[:, -1:], test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print('İkinci tahmin sonucu: ', y_pred)

# Tahmin sonuçlarını görselleştirme (Opsiyonel)
plt.scatter(x_test.index, y_test, color='red')
plt.plot(x_test.index, y_pred, color='blue')
plt.title('Tahmin Sonuçları')
plt.xlabel('Gözlem Numarası')
plt.ylabel('Oynama Durumu')
plt.show()