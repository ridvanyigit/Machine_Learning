import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükle
veriler = pd.read_csv("veriler.csv")

# Kategorik değişkenleri One-Hot Encoding ile dönüştür
veriler = pd.get_dummies(veriler, columns=['ulke'], drop_first= True)

# Bağımsız değişkenler ve hedef değişkeni ayır
X = veriler.drop('cinsiyet', axis=1)
y = veriler['cinsiyet']

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Random Forest modelini oluştur ve eğit
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# Test verisiyle tahmin yap
y_pred = model.predict(X_test)

# Modeli test verisiyle değerlendir
accuracy = accuracy_score(y_test, y_pred)
print("Modelin doğruluk skoru:", accuracy)

# Karmaşıklık matrisini oluştur
cm = confusion_matrix(y_test, y_pred)
print('karmasiklik matrisi:')
print(cm)

# Karmaşıklık matrisini görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Erkek', 'Kadın'], yticklabels=['Erkek', 'Kadın'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Karmaşıklık Matrisi')
plt.show()


