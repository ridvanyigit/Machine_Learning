import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


                   ######   Veri ön isleme   #######


                     # 1.ADIM : Veri yükleme

veriler = pd.read_csv('aylara_gore_satislar.csv')                            
print(veriler)                                   

aylar = veriler[['Aylar']]                                                    
#print(aylar)
                          
satislar = veriler[['Satislar']]                                               
#print(satislar1)

             

                    # 2. ADIM : Veri setinin Egitim ve Test kümelerine bölünmesi

from sklearn.model_selection import train_test_split                           

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)


                      # 3. ADIM : Verilerin Ölceklendirilmesi
                      
from sklearn.preprocessing import StandardScaler                               

sc = StandardScaler()                                                         

X_train = sc.fit_transform(x_train)                                      
X_test = sc.fit_transform(x_test)                                              

Y_train = sc.fit_transform(y_train)                                            
Y_test = sc.fit_transform(y_test)                                              

'''Buraya kadar olan islemler CRISP DM Yönteminde Veri Ön Isleme (Data Preprocessing) asamasi ile ile ilgiliydi.
Bundan sonraki islemlerde CRISP DM Yönteminde olan MODELING asamasi islemleri olacak.'''


                     # 4. ADIM : Model Insasi (Linear Regression)
                     
from sklearn.linear_model import LinearRegression

lr = LinearRegression()                                                        

lr.fit(x_train, y_train)                                                      
tahmin = lr.predict(x_test)                                                    


                    # 5. ADIM : DataFrame'lerin Grafiksel olarak gösterilmesi
                    
x_train = x_train.sort_index()                                                
y_train = y_train.sort_index()                                                 

plt.plot(x_train, y_train)                                                    
plt.plot(x_test,lr.predict(x_test))                                           

plt.title('Aylara Göre SAtislar')                                           
plt.xlabel('Aylar')                                                          
plt.ylabel('Satislar')                                                        




