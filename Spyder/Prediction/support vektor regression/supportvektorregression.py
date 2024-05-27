import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Veri Yükleme
data = pd.read_csv('maaslar.csv')

# Verileri Ayırma (Slicing)
X = data[['Egitim Seviyesi']].values
y = data[['maas']].values

# Verilerin Ölçeklendirilmesi
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# SVR Modeli Oluşturma ve Eğitme
model = SVR(kernel='rbf')
model.fit(X_scaled, y_scaled)

# Tahmin ve Görselleştirme
plt.scatter(X_scaled, y_scaled, color='red', label='Gerçek Veriler')
plt.plot(X_scaled, model.predict(X_scaled), color='blue', label='SVR Tahminleri')
plt.title('Eğitim Seviyesine Göre Maaş Tahminleri')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş (Scaled)')
plt.legend()
plt.show()

# Yeni Veri ile Tahmin Yapma
education_level_6 = scaler_X.transform([[6]])
education_level_8 = scaler_X.transform([[8]])

salary_pred_6 = scaler_y.inverse_transform(model.predict(education_level_6).reshape(-1, 1))
salary_pred_8 = scaler_y.inverse_transform(model.predict(education_level_8).reshape(-1, 1))

print(f'Eğitim Seviyesi 6 için tahmin edilen maaş: {salary_pred_6[0][0]}')
print(f'Eğitim Seviyesi 8 için tahmin edilen maaş: {salary_pred_8[0][0]}')

# Model Performansını Değerlendirme
r2_value = r2_score(y_scaled, model.predict(X_scaled))
print(f'SVR Model R2 Değeri: {r2_value}')
