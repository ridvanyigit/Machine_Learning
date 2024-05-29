import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veriyi yükleme
df = pd.read_csv('veriler.csv')

# Kategorik veriyi sayısal veriye dönüştürme
le = LabelEncoder()
df['ulke'] = le.fit_transform(df['ulke'])
df['cinsiyet'] = le.fit_transform(df['cinsiyet'])

# Özellikleri ve hedef değişkeni belirleme
X = df[['ulke', 'boy', 'kilo', 'yas']]
y = df['cinsiyet']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes modelini oluşturma
model = GaussianNB()

# Modeli eğitme
model.fit(X_train, y_train)

# Test seti ile tahmin yapma
y_pred = model.predict(X_test)

# Modelin performansını değerlendirme
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Karışıklık matrisi hesaplama
cm = confusion_matrix(y_test, y_pred)

# Karışıklık matrisini görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['e', 'k'], yticklabels=['e', 'k'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Eğitim ve test setlerinin dağılımını görselleştirme
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_train['boy'], y=X_train['kilo'], hue=y_train, palette='coolwarm')
plt.title('Training Set')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test['boy'], y=X_test['kilo'], hue=y_pred, palette='coolwarm')
plt.title('Test Set Predictions')

plt.show()
