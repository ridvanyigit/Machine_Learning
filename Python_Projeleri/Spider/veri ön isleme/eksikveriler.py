import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


                   ######   Veri ön isleme   #######


                     # 1.ADIM : Veri yükleme

veriler = pd.read_csv('eksikveriler.csv')                                      # Bu satirda üzerinde islem yapmak istedigimiz dosyayi Python'a yüklüyoruz.
print(veriler)
                          
                     # 2. ADIM : Eksik veriler (Missing Values)
                             
from sklearn.impute import SimpleImputer                                       # SimpleImputer sinifi eksik verilerin tespit edilmesi(missing_values= np.nan), onlarin belirlenen bir stratejiye(strategy= "mean") göre doldurulmasi islemlerini yapar.

yas = veriler.iloc[:,1:4].values                                               # Bu satirda veriler veri kümesindeki 1. ve 4. indeksler arasındaki (4. indeks dahil değil) sütunları seçiyor ve bu sütunların değerlerini bir NumPy dizisi olarak yas değişkenine atıyoruz. Yani; bir bütün olan dosyanin icinden bu sütunu ayirip onu tek bir sütun olarak isliyoruz.
#print(yas)

ulke = veriler.iloc[:,0:1].values                                              # Bu satirda verilerin ilk sütununu (0 indeks) seçiyoruz ve bu sütunun değerlerini bir NumPy dizisi olarak ulke değişkenine atıyoruz. Yani; bir bütün olan dosyanin icinden bu sütunu ayirip onu tek bir sütun olarak isliyoruz.
#print(ulke)

cinsiyet = veriler.iloc[:,-1].values                                           # Bu satirda verilerin son sütununu (-1 indeks) seçiyoruz ve bu sütunun değerlerini bir NumPy dizisi olarak 'cinsiyet' değişkenine atıyoruz. Yani; bir bütün olan dosyanin icinden bu sütunu ayirip onu tek bir sütun olarak isliyoruz.
#print(cinsiyet)

imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')              # burada eksik olarak belirlenen degeri sütundaki diger degerlerin ortalamasi stratejisine göre dolduruyoruz.

imputer = imputer.fit(yas[:,1:4])                                              #fit() işlemi istatistikleri hesaplar ve bunları kullanarak bir doldurma stratejisi belirler.
yas[:,1:4] = imputer.transform(yas[:,1:4])                                     #transform() işlemi, bir doldurma stratejisi belirlendikten sonra, bu stratejiyi kullanarak eksik değerleri doldurur.
#print(yas)                                                                    #yas[:, 1:4] = imputer.fit_transform(yas[:, 1:4]) olarak da bu iki satiri tek satirda yazmak mümkündür.



                       # 3. ADIM : Kategorik Veriler 
                      
from sklearn import preprocessing                                              # Preprocessing sınıfı; Veri önişleme, veri setinin temizlenmesi, dönüştürülmesi ve hazırlanması işlemlerini içerir.

le = preprocessing.LabelEncoder()                                              # LabelEncoder; kategorik verileri sırasıyla 0, 1, 2 gibi sayısal etiketlere dönüştürür. 
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
#print(ulke) 

ohe = preprocessing.OneHotEncoder()                                            # OneHotEncoder; kategorik verileri sıralamadan ziyade, her bir kategoriye ait özel bir sütun oluşturarak kategorik verileri ikili (0 ve 1) vektörlerine dönüştürür.
ulke = ohe.fit_transform(ulke).toarray()
#print(ulke) 



                      #4. ADIM : Numpy dizilerini DataFrame'e dönüstürme

sonuc = pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])       # Bu kod, ulke adlı veri setinden 'fr, tr, us' sütunları oluşturarak yeni bir DataFrame oluşturur.
#print(sonuc)

sonuc2 = pd.DataFrame(data=yas, index=range(22),columns=['boy','kilo','yas'])  # Bu kod, yas adlı veri setinden 'boy, kilo, yas' sütunları oluşturarak yeni bir DataFrame oluşturur.
#print(sonuc2)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22),columns=['cinsiyet'])     # Bu kod, cinsiyet adlı veri setinden 'cinsiyet' sütununu alarak yeni bir DataFrame oluşturur.
#print(sonuc3)


                      #5. ADIM : DataFrame'leri birlestirme
                      
s = pd.concat([sonuc,sonuc2], axis=1)
                                                                               
s2 = pd.concat([s,sonuc3], axis=1)
                                                                               # Sonuc olarak: ilk basta iceriginde eksik verilerin bulundugu 'eksikveriler' adli dosyayi indeks sayili satirlari olan ve sütun basliklari olan 'birlesik_dataframe' veri setine dönüstürdük.                         
print(s2)                                                                      # birlesik_dataframe = pd.concat([sonuc1,sonuc2,sonuc3], axis=1)  olarark tek bir kod satiri olarak da sonucu yazabilirdik. Tek satirda yazmamamizin amaci satirlara bölme islemine uygun hale getirmek.




                      # 6. ADIM : Veri kümesinin Egitim ve Test icin bölünmesi

from sklearn.model_selection import train_test_split                           # train_test_split fonksiyonu, veri setini; öğrenme için kullanılan eğitim (train) verisi ve modelin performansını değerlendirmek için kullanılan test verisi olarak iki ayrı parçaya böler.

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)     # Burada s egitim sütunlari olarak, sonuc3 ise test sütunlari olarak belirtilir. sonuc1 ile sonuc2'yi bir öndeki asamada s1'in icine aktarmamizin sebebi buydu. Yani train verileri olarak kullanmak icin 0.67 oraninda verileri ayni ifade altina aldik.


                      # 7. ADIM: Verilerin Ölceklendirilmesi
                      
from sklearn.preprocessing import StandardScaler                               # yukarida elde ettigimiz ölcek anlaminda birbirinden oldukca farkli büyüklükte degerler iceren egitim ve test kümelerini, makinenin anlayabilecegi birbirine büyüklükteki verilere ölceklendirdik.

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)






