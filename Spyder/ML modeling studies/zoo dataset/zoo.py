import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Veri setini yükle
df = pd.read_csv("zoo.csv")
data = df.copy()

# Kategorik sütunu veri kümesinden çıkar
data.drop('animal_name', axis=1, inplace=True)

# Bağımsız değişkenleri ve hedef değişkeni ayır
X = data.drop(['class_type'], axis=1)
y = data['class_type']

# Sınıf dağılımını kontrol et
print(y.value_counts())

# Veriyi eğitim ve test setlerine stratified split ile böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Sınıflandırma algoritmalarını tanımla
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

# Her bir sınıflandırıcı için model oluştur, eğit, tahmin yap ve değerlendir
for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    # Model performansını değerlendir
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred, zero_division=1)}")
    
    # Hata matrisini oluştur ve görselleştir
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} Confusion Matrix:")
    print(cm)
