import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Veri Yükleme
data = pd.read_csv('maaslar.csv')

# Verileri Ayırma (Slicing)
X = data[['Egitim Seviyesi']].values
y = data[['maas']].values

# Verilerin Eğitim ve Test Kümesine Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Decision Tree Regresyon Modeli Oluşturma ve Eğitme
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X_train, y_train)

# Tahmin ve Görselleştirme
plt.scatter(X, y, color='red', label='Gerçek Veriler')
plt.plot(X, dt_regressor.predict(X), color='blue', label='Decision Tree Tahminleri')
plt.title('Eğitim Seviyesine Göre Maaş Tahminleri (Decision Tree Regresyon)')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.legend()
plt.show()

# Yeni Veri ile Tahmin Yapma
education_level_6 = [[6]]
education_level_8 = [[8]]

salary_pred_6 = dt_regressor.predict(education_level_6)
salary_pred_8 = dt_regressor.predict(education_level_8)

print(f'Eğitim Seviyesi 6 için tahmin edilen maaş: {salary_pred_6[0]}')
print(f'Eğitim Seviyesi 8 için tahmin edilen maaş: {salary_pred_8[0]}')

# Model Performansını Değerlendirme
y_pred = dt_regressor.predict(X_test)
r2_value = r2_score(y_test, y_pred)
print(f'Decision Tree Regresyon Modeli R2 Değeri: {r2_value}')

