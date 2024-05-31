import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
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

# Sınıflandırma algoritmalarını bir liste olarak tanımla
classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(),
    SVC(kernel='linear'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB()
]

# Her bir sınıflandırıcı için işlemleri tekrarla
for classifier in classifiers:
    # Model oluşturma ve eğitme
    classifier_name = classifier.__class__.__name__
    print(f"Training {classifier_name}...")
    classifier.fit(X_train, y_train)
    
    # Test seti üzerinde tahmin yapma
    y_pred = classifier.predict(X_test)
    
    # Doğruluk değerini hesapla
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{classifier_name} Accuracy: {accuracy:.2f}")
    
    # Karışıklık matrisini hesapla
    cm = confusion_matrix(y_test, y_pred)
    print(f'{classifier_name} Confusion Matrix:')
    print(cm)
    
    # Karışıklık matrisini görselleştir
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot(cmap='Blues')
    disp.ax_.set_title(f"{classifier_name} - Confusion Matrix\nAccuracy: {accuracy:.2f}")
    
    # Sınıflandırma raporunu hesapla ve ekrana yazdır
    report = classification_report(y_test, y_pred)
    print(f"{classifier_name} Classification Report:")
    print(report)
    print("\n")
    
