import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Veri yükleme
veriler = pd.read_csv('veriler.csv')

# Yas verilerini al
Yas = veriler.iloc[:, 1:4].values

# Kategorik -> Numeric dönüşüm (Ülke)
ulke = veriler.iloc[:, 0:1].values
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

# Kategorik -> Numeric dönüşüm (Cinsiyet)
cinsiyet = veriler.iloc[:, -1].values
c = veriler.iloc[:, -1:].values
c[:, -1] = le.fit_transform(veriler.iloc[:, -1])
c = ohe.fit_transform(c).toarray()

# Numpy dizilerini DataFrame'e dönüştürme
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas'])
sonuc3 = pd.DataFrame(data=c[:, :1], index=range(22), columns=['cinsiyet'])

# DataFrame'leri birleştirme
s = pd.concat([sonuc, sonuc2], axis=1)
s2 = pd.concat([s, sonuc3], axis=1)

# Verilerin eğitim ve test için bölünmesi
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

# Çoklu doğrusal regresyon uygulama
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# 'boy' değişkenini ayırma
boy = s2.iloc[:, 3:4].values

# Geriye kalan veriler
sol = s2.iloc[:, :3]
sag = s2.iloc[:, 4:]
veri = pd.concat([sol, sag], axis=1)

# Verilerin eğitim ve test için bölünmesi (boy tahmini)
x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train, y_train)

y_pred = r2.predict(x_test)

# Başarıyı ölçme (p-değeri ölçümü)

# Backward Elimination (Geriye Doğru Eleme)
X = np.append(arr=np.ones((22, 1)).astype(int), values=veri, axis=1)

# İlk model
X_l = veri.iloc[:, [0, 1, 2, 3, 4, 5]].values
r_ols = sm.OLS(endog=boy, exog=X_l)
r = r_ols.fit()
print(r.summary())

# En yüksek p-değerine sahip sütunu kaldırma (4. indeks)
X_l = veri.iloc[:, [0, 1, 2, 3, 5]].values
r_ols = sm.OLS(endog=boy, exog=X_l)
r = r_ols.fit()
print(r.summary())