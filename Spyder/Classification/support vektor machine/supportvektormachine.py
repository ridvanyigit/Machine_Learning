import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Veriyi yükleme
df = pd.read_csv('veriler.csv')

# Veriyi inceleme
print(df.head())

# Ülke sütununu etiketleme
df['ulke'] = df['ulke'].astype('category').cat.codes

# Cinsiyet sütununu etiketleme
df['cinsiyet'] = df['cinsiyet'].map({'e': 0, 'k': 1})

# Özellikler ve hedef değişkeni ayırma
X = df[['ulke', 'boy', 'kilo', 'yas']]
y = df['cinsiyet']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM modelini oluşturma
svm_model = SVC(kernel='rbf')  # Kernel türünü isteğe göre değiştirebilirsiniz
svm_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = svm_model.predict(X_test)

# Model performansını değerlendirme
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Opsiyonel: Modeli kaydetme
import joblib
joblib.dump(svm_model, 'svm_model.pkl')
