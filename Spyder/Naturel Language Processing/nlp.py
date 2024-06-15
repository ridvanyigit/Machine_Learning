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

# Load the dataset
reviews = pd.read_csv('Restaurant_Reviews2.csv')

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Compilation list to store processed reviews
corpus = []

# Processing each review
for i in range(len(reviews)):
    # Remove punctuation and convert to lowercase
    review = re.sub(r'[^a-zA-Z\s]', '', reviews['Review'][i])
    review = review.lower()
    review = review.split()
    
    # Remove stopwords and perform stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    # Join the stemmed words back into a single string
    review = ' '.join(review)
    
    # Append to the corpus list
    corpus.append(review)

# Print the first five processed reviews for verification
for i in range(5):
    print(f"Example {i+1}: {corpus[i]}")

# =============================================================================
# Feature Extraction - Öznitelik Çıkarımı
# =============================================================================

cv = CountVectorizer(max_features=2000)

X = cv.fit_transform(corpus).toarray()
y = reviews.iloc[:, 1].values

# =============================================================================
# Machine Learning - Makine Öğrenmesi
# =============================================================================

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

