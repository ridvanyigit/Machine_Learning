import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Veriyi yükle
data = pd.read_csv('veriler.csv')

# Bağımlı ve bağımsız değişkenleri ayır
X = data[['boy', 'kilo', 'yas']]
y = data['cinsiyet']

# Cinsiyet verisini sayısal hale getir (k: 0, e: 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Lojistik Regresyon modelini oluştur ve eğit
model = LogisticRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# Modelin doğruluğunu değerlendir
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Karışıklık matrisini hesapla
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

# Karışıklık matrisini görselleştir
sns.heatmap(conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels= label_encoder.classes_, 
            yticklabels= label_encoder.classes_
            )
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
