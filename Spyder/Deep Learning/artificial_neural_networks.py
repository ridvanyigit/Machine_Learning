import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense


# Veri yükleme
data = pd.read_csv('Churn_Modelling.csv')

# Veri ön işleme
X = data.iloc[:, 3:13].values
Y = data.iloc[:, 13].values

# Kategorik verileri sayısal formata dönüştürme
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

le2 = LabelEncoder()
X[:, 2] = le2.fit_transform(X[:, 2])

# One-Hot Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Verilerin eğitim ve test için bölünmesi
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

# Verilerin ölçeklenmesi
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Yapay Sinir Ağı modeli oluşturma
classifier = Sequential()

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=12))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=50)

# Tahminler ve confusion matrix hesaplama
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

