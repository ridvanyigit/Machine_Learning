import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Veri setini yükleme
veriler = pd.read_csv('eksikveriler.csv')

# Eksik verilerin düzeltilmesi
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
yas = veriler[['boy', 'kilo', 'yas']].values
imputer = imputer.fit(yas)
yas = imputer.transform(yas)

# Kategorik verilerin işlenmesi
ulke = veriler[['ulke']].values
le = LabelEncoder()
ulke = le.fit_transform(ulke.ravel())
ohe = OneHotEncoder()
ulke = ohe.fit_transform(ulke.reshape(-1, 1)).toarray()

# DataFrame oluşturma
ulke_df = pd.DataFrame(data=ulke, columns=['France', 'Turkey', 'USA'])
yas_df = pd.DataFrame(data=yas, columns=['boy', 'kilo', 'yas'])
cinsiyet_df = pd.DataFrame(data=veriler['cinsiyet'], columns=['cinsiyet'])
birlesik_df = pd.concat([ulke_df, yas_df, cinsiyet_df], axis=1)

# Veri setinin eğitim ve test için bölünmesi
X = birlesik_df.drop('cinsiyet', axis=1).values
y = birlesik_df['cinsiyet'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Verilerin ölçeklendirilmesi
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Sonuçlar
print("Eğitim veri seti:")
print(X_train)
print("\nTest veri seti:")
print(X_test)

