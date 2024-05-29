import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Veri setini yükle
df = pd.read_excel('iris.xls')
data = df.copy()
print(data.head())

# Özellikler ve hedef değişkeni ayır
X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = data['iris']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest modelini oluştur ve eğit
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test seti üzerinde tahminler yap
y_pred = model.predict(X_test)

# Doğruluk değerini hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy:.2f}")

# Karışıklık matrisini hesapla
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Karışıklık matrisini görselleştir
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
disp.ax_.set_title(f"Random Forest - Confusion Matrix\nAccuracy: {accuracy:.2f}")

# Sınıflandırma raporunu hesapla ve ekrana yazdır
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

