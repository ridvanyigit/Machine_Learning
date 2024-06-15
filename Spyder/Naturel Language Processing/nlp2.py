import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# =============================================================================
# Preprocessing - Veri Ön İşleme
# =============================================================================

# Veri setini yükle
yorumlar = pd.read_csv('Restaurant_Reviews2.csv')

# NLTK stopwords indirme
nltk.download('stopwords')

# Porter Stemmer örneği oluşturma
ps = PorterStemmer()

# Derlem listesi oluşturma
derlem = []

# Her bir yorum için işlemleri yapma
for i in range(len(yorumlar)):
    # İmla işaretlerini kaldırma ve küçük harflere dönüştürme
    yorum = re.sub(r'[^a-zA-Z\s]', '', yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    
    # Stopwords'leri kaldırma ve kelime köklerini alarak stemleme
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    
    # Kök haline getirilmiş kelimeleri tekrar birleştirme
    yorum = ' '.join(yorum)
    
    # Derlem listesine ekleme
    derlem.append(yorum)

# Sonuçları kontrol etmek için ilk beş öğeyi yazdırma
for i in range(5):
    print(f"Örnek {i+1}: {derlem[i]}")

# =============================================================================
# Feature Extraction - Öznitelik Çıkarımı
# =============================================================================

cv = CountVectorizer(max_features=2000)

X = cv.fit_transform(derlem).toarray()
y = yorumlar.iloc[:, 1].values

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# Machine Learning - Makine Öğrenmesi
# =============================================================================

# Gaussian Naive Bayes modelini oluşturma ve eğitme
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Test seti ile tahmin yapma
y_pred = gnb.predict(X_test)

# Sonuçları değerlendirme
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy:.2f}")

