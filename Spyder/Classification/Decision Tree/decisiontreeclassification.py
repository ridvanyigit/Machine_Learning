import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Verileri oku
data = pd.read_csv('veriler.csv')

# One-Hot Encoding kullanarak kategorik değişkenleri dönüştür
data = pd.get_dummies(data, columns=['ulke', 'cinsiyet'], drop_first=True)

# Bağımsız değişkenler ve hedef değişkeni ayır
X = data.drop('cinsiyet_k', axis=1)  # cinsiyet_k kategorik değişkeni k olarak kodlandığından drop_first=True kullanıldığında bu sütunu düşürmek gerekir.
y = data['cinsiyet_k']  # 'cinsiyet_k' hedef değişkenidir

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Karar ağacı sınıflandırıcı oluştur
model = DecisionTreeClassifier()

# Modeli eğit
model.fit(X_train, y_train)

# Test verisiyle tahmin yap
y_pred = model.predict(X_test)

# Modelin doğruluğunu değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f'Doğruluk: {accuracy}')

# Detaylı sınıflandırma raporu
print(classification_report(y_test, y_pred))

# Karışıklık matrisi
cm = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:")
print(cm)

# Karar ağacı yapısını yazdırma
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)

# Karar ağacını görselleştirme
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=list(X.columns), class_names=['e', 'k'], filled=True)
plt.show()

