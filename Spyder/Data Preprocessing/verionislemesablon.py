import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Verilerin yüklenmesi
veriler = pd.read_csv('veriler.csv')

# Yas verileri
yas = veriler[['boy', 'kilo', 'yas']].values

# Ulke verileri
ulke = veriler[['ulke']].values
ulke_encoder = preprocessing.LabelEncoder()
ulke[:, 0] = ulke_encoder.fit_transform(veriler['ulke'])
ulke_onehot = preprocessing.OneHotEncoder()
ulke = ulke_onehot.fit_transform(ulke).toarray()

# Cinsiyet verileri
cinsiyet = veriler[['cinsiyet']].values
cinsiyet_encoder = preprocessing.LabelEncoder()
cinsiyet[:, 0] = cinsiyet_encoder.fit_transform(veriler['cinsiyet'])
cinsiyet_onehot = preprocessing.OneHotEncoder()
cinsiyet = cinsiyet_onehot.fit_transform(cinsiyet).toarray()

# Numpy dizilerini DataFrame'e dönüştürme
ulke_df = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
yas_df = pd.DataFrame(data=yas, index=range(22), columns=['boy', 'kilo', 'yas'])
cinsiyet_df = pd.DataFrame(data=cinsiyet[:, :1], index=range(22), columns=['cinsiyet'])

# DataFrame birleştirme işlemi
birlesik_veriler = pd.concat([ulke_df, yas_df, cinsiyet_df], axis=1)

# Verilerin eğitim ve test için bölünmesi
X_train, X_test, y_train, y_test = train_test_split(birlesik_veriler.drop('cinsiyet', axis=1), birlesik_veriler['cinsiyet'], test_size=0.33, random_state=0)

