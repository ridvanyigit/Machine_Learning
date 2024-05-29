import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Veri Yükleme
data = pd.read_csv('maaslar.csv')

# Verileri Ayırma (Slicing)
X = data[['Egitim Seviyesi']].values
y = data[['maas']].values

# Verilerin Eğitim ve Test Kümesine Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Polynomial Özelliklerin Eklenmesi
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)

# Polynomial Regresyon Modeli Oluşturma ve Eğitme
polynomial_reg = LinearRegression()
polynomial_reg.fit(X_poly, y_train)

# Tahmin ve Görselleştirme
plt.scatter(X, y, color='red', label='Gerçek Veriler')
plt.plot(X, polynomial_reg.predict(poly_reg.transform(X)), color='blue', label='Polynomial Regresyon Tahminleri')
plt.title('Eğitim Seviyesine Göre Maaş Tahminleri (Polynomial Regresyon)')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.legend()
plt.show()

# Yeni Veri ile Tahmin Yapma
education_level_6 = [[6]]
education_level_8 = [[8]]

salary_pred_6 = polynomial_reg.predict(poly_reg.transform(education_level_6))
salary_pred_8 = polynomial_reg.predict(poly_reg.transform(education_level_8))

print(f'Eğitim Seviyesi 6 için tahmin edilen maaş: {salary_pred_6[0][0]}')
print(f'Eğitim Seviyesi 8 için tahmin edilen maaş: {salary_pred_8[0][0]}')

# Model Performansını Değerlendirme
X_test_poly = poly_reg.transform(X_test)
y_pred = polynomial_reg.predict(X_test_poly)
r2_value = r2_score(y_test, y_pred)
print(f'Polynomial Regresyon Modeli R2 Değeri: {r2_value}')
