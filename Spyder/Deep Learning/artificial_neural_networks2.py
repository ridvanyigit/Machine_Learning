import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
data = pd.read_csv('Churn_Modelling.csv')

# Özellikler ve etiketlerin ayrılması
X = data.iloc[:, 3:-1].values  # Sütunlar 3'ten başlayarak son sütun hariç alınır
y = data.iloc[:, -1].values  # Son sütun

# Kategorik verilerin kodlanması
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # Geography
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])  # Gender

# OneHotEncoder kullanarak Geography sütununu dönüştürme
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]  # Dummy değişken tuzağından kaçınmak için ilk sütunu çıkar

# Verilerin eğitim ve test setlerine ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Verilerin ölçeklendirilmesi
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN modelini oluşturma
model = Sequential()

# İlk gizli katman
model.add(Dense(units=6, activation='relu', input_dim=11))

# İkinci gizli katman
model.add(Dense(units=6, activation='relu'))

# Çıkış katmanı
model.add(Dense(units=1, activation='sigmoid'))

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, batch_size=10, epochs=100, validation_split=0.2)

# Eğitim ve doğrulama kayıplarının görselleştirilmesi
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluğunun görselleştirilmesi
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Tahminler yapma
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Karışıklık matrisi ve doğruluk skoru
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(f'Doğruluk: {accuracy}')

# Karışıklık matrisinin ısı haritası
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

