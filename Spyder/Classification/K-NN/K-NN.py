import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Veriyi yükleme
data = pd.read_csv('veriler.csv')

# Özellikler ve etiketler
X = data[['boy', 'kilo', 'yas']]
y = data['cinsiyet']

# Cinsiyet etiketini sayısal değerlere çevirme
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # 'e' -> 0, 'k' -> 1

# Veriyi normalize etme
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Çapraz doğrulama ve model performansı
models = {
    'K-NN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

for model_name, model in models.items():
    # Hiperparametre optimizasyonu
    if model_name == 'K-NN':
        param_grid = {'n_neighbors': range(1, 11)}
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == 'SVM':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'rbf']
        }
    
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # En iyi modeli kullanarak test verisi üzerinde tahmin yapma
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Modelin doğruluğunu değerlendirme
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} doğruluğu: {accuracy * 100:.2f}%")
    print(f"En iyi parametreler: {grid_search.best_params_}")
    
    # Karışıklık matrisi oluşturma ve görselleştirme
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot()
    plt.title(f'{model_name} Karışıklık Matrisi')
    plt.show()
    
    