# PCA (Principal Component Analysis)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Veri setini yükleyin
data = pd.read_csv('Wine.csv')

# Özellikler ve hedef değişkeni ayırın
X = data.iloc[:, :-1]  # Özellikler
y = data.iloc[:, -1]   # Hedef değişken (Customer_Segment)

# Veriyi standartlaştırın
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA uygulayın
pca = PCA(n_components=2)  # İlk 2 bileşeni kullanarak
X_pca = pca.fit_transform(X_scaled)

# Veri setini eğitim ve test olarak ayırın
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Lojistik Regresyon modeli oluşturun ve eğitin
model = LogisticRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapın
y_pred = model.predict(X_test)

# Sonuçları değerlendirin
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# PCA sonuçlarını görselleştirin
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Wine Dataset')
plt.colorbar(scatter, label='Customer Segment')
plt.show()
