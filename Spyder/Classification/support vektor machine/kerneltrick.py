import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Veriyi yükleme
df = pd.read_csv('veriler.csv')

# Ülke sütununu etiketleme
df['ulke'] = df['ulke'].astype('category').cat.codes

# Cinsiyet sütununu etiketleme
df['cinsiyet'] = df['cinsiyet'].map({'e': 0, 'k': 1})

# Özellikler ve hedef değişkeni ayırma
X = df[['ulke', 'boy', 'kilo', 'yas']]
y = df['cinsiyet']

# Stratified split kullanarak eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)

# Veriyi standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Farklı çekirdek fonksiyonlarını kullanarak SVM modelleri oluşturma
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    print(f'\nKernel: {kernel}')
    
    # SVM modelini oluşturma
    svm_model = SVC(kernel= kernel)
    svm_model.fit(X_train, y_train)
    
    # Test seti üzerinde tahmin yapma
    y_pred = svm_model.predict(X_test)
    
    # Model performansını değerlendirme
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division= 0))
    
    # Confusion matrix görselleştirme
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot= True, fmt='d', cmap='Blues', xticklabels=['e', 'k'], yticklabels= ['e', 'k'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {kernel} Kernel')
    plt.show()
    
    # Sınıflandırma sınırlarını görselleştirme (2D projeksiyon)
    X_vis = X_train[:, :2]  # İlk iki özelliği kullanarak
    y_vis = y_train
    
    # Modeli yeniden eğitme (sadece ilk iki özellik)
    svm_model_2d = SVC(kernel= kernel)
    svm_model_2d.fit(X_vis, y_vis)
    
    # Meshgrid oluşturma
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Meshgrid üzerindeki her noktayı sınıflandırma
    Z = svm_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('First feature')
    plt.ylabel('Second feature')
    plt.title(f'SVC with {kernel} kernel (First two features)')
    plt.show()
    
