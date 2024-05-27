import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

###### Veri ön işleme #######

# 1. ADIM: Veri yükleme
veriler = pd.read_csv('aylara_gore_satislar.csv')
print(veriler)

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

# 2. ADIM: Veri setinin Eğitim ve Test kümelerine bölünmesi
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

# 3. ADIM: Verilerin Ölçeklendirilmesi
sc = StandardScaler()

# x_train ve x_test ayrı ayrı ölçeklendirilmelidir
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)  # fit_transform yerine transform kullanılır

# y_train ve y_test ayrı ayrı ölçeklendirilmelidir
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(y_train)
Y_test = sc_y.transform(y_test)  # fit_transform yerine transform kullanılır

# 4. ADIM: Model İnşası (Linear Regression)
model = LinearRegression()
model.fit(x_train, y_train)  # Orijinal ölçeklendirilmemiş veriler kullanılır
tahmin = model.predict(x_test)

# 5. ADIM: DataFrame'lerin Grafiksel olarak gösterilmesi
x_train_sorted = x_train.sort_index()
y_train_sorted = y_train.sort_index()

plt.figure(figsize=(10, 6))
plt.plot(x_train_sorted, y_train_sorted, label='Training Data')
plt.plot(x_test, tahmin, label='Predicted Test Data', color='red')
plt.title('Aylara Göre Satışlar')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
plt.legend()
plt.show()