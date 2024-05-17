import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


                   ######   Veri ön isleme   #######


                     # 1.ADIM : Veri yükleme

veriler = pd.read_csv('aylara_gore_satislar.csv')                              # veri setini DataFrame'e dönüstürdük
print(veriler)                                   

aylar = veriler[['Aylar']]                                                     # veri setindeki 'Aylar' adindaki sütununu kodlamada DataFrame olarak kullanmak üzere 'aylar' olarak isimlendirdik
#print(aylar)
                          
satislar = veriler[['Satislar']]                                               # veri setindeki 'Satislar' adindaki sütununu kodlamada DataFrame olarak kullanmak üzere 'satislar' olarak isimlendirdik
#print(satislar1)

             

                    # 2. ADIM : Veri setinin Egitim ve Test kümelerine bölünmesi

from sklearn.model_selection import train_test_split                           # train_test_split fonksiyonu, veri setini; öğrenme için kullanılan eğitim (train) verisi ve modelin performansını değerlendirmek için kullanılan test verisi olarak iki ayrı parçaya böler.

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)


                      # 3. ADIM : Verilerin Ölceklendirilmesi
                      
from sklearn.preprocessing import StandardScaler                               # yukarida elde ettigimiz ölcek anlaminda birbirinden oldukca farkli büyüklükte degerler iceren egitim ve test kümelerini, Makine Ögrenimi araliklarina ölceklendirecegiz.

sc = StandardScaler()                                                          # standart ölceklendirme islemini uygulayip sc adinda bir ifadeye atadik

X_train = sc.fit_transform(x_train)                                            # ölceklendirme islemini sc üzerinden x_train DataFrame'e uyguluyoruz
X_test = sc.fit_transform(x_test)                                              # ölceklendirme islemini sc üzerinden x_test DataFrame'e uyguluyoruz

Y_train = sc.fit_transform(y_train)                                            # ölceklendirme islemini sc üzerinden y_train DataFrame'e uyguluyoruz
Y_test = sc.fit_transform(y_test)                                              # ölceklendirme islemini sc üzerinden x_test DataFrame'e uyguluyoruz

'''Buraya kadar olan islemler CRISP DM Yönteminde Veri Ön Isleme (Data Preprocessing) asamasi ile ile ilgiliydi.
Bundan sonraki islemlerde CRISP DM Yönteminde olan MODELING asamasi islemleri olacak.'''


                     # 4. ADIM : Model Insasi (Linear Regression)
                     
from sklearn.linear_model import LinearRegression

lr = LinearRegression()                                                        # Linear Regresyon hesaplayabilme yetenegini 'lr' adinda olusturdugumuz modele aktardik.

lr.fit(x_train, y_train)                                                       # lr modelini x_train ve y_train egitim verileri ile fit() fonksiyonunu kullanarak egittik
tahmin = lr.predict(x_test)                                                    # egitilmis olan lr modelini x_test verilerine uygulayip y_test verilerini tahmin ettirdik.


                    # 5. ADIM : DataFrame'lerin Grafiksel olarak gösterilmesi
                    
x_train = x_train.sort_index()                                                 # olusturmak istedigimiz grafiklerde yataydaki (X) indekslerin sirali olarak artmalari icin sort_index() fonksiyonunu kullandik
y_train = y_train.sort_index()                                                 # olusturmak istedigimiz grafiklerde dikeydeki (Y) indekslerin sirali olarak artmalari icin sort_index() fonksiyonunu kullandik

plt.plot(x_train, y_train)                                                     # verileri noktalar halinde bir grafikte gösterir
plt.plot(x_test,lr.predict(x_test))                                            # olusturulan grafikteki noktalara göre bir dogrusal regresyon cizgisi cizer

plt.title('Aylara Göre SAtislar')                                              # grafige baslik atadik
plt.xlabel('Aylar')                                                            # X yatayini isimlendirdik
plt.ylabel('Satislar')                                                         # Y dikeyini isimlendirdik




