import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm



veriler = pd.read_csv('odev_tenis.csv')
print('veriler:\n',veriler)




#encoder: Kategorik -> Numeric
veriler2 = veriler.apply(LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]

ohe = OneHotEncoder()
c = ohe.fit_transform(c).toarray()
#print(c)

havadurumu = pd.DataFrame(data= c, index= range(14), columns= ['o','r','s'])
sonveriler = pd.concat([havadurumu, veriler.iloc[:,1:3]], axis= 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis= 1)
print('\nSon Veriler: \n',sonveriler)



#verilerin egitim ve test icin bolunmesi
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)



#coklu dogrusl regresyon uygulama
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print('birinci tahmin sonucu: ', y_pred)

             #Modelin basarisinin degerlendirilmesi.( Olasilik degeri (p_degeri) ölcümü )
             
             
#Backward Elimination ( Geriye Dogru Eleme )
X = np.append(arr= np.ones((14,1)).astype(int), values= sonveriler.iloc[:,:-1], axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog= X.iloc[:,-1:], exog= X_l)
r = r_ols.fit()                
print('\nBirinci Degerlendirme Raporu: ',r.summary())             

#yukaridaki en yüksek p degerine sahip olan 0. index'teki sütunu sildik
X = np.append(arr= np.ones((14,1)).astype(int), values= sonveriler.iloc[:,:-1], axis=1)
X_l = sonveriler.iloc[:,[1,2,3,4,5]].values
r_ols = sm.OLS(endog= sonveriler.iloc[:,-1:], exog= X_l)
r = r_ols.fit()                
print('\nIkinci Degerlendirme Raporu: ',r.summary()) 





#Buraya kadarki raporlara bakildigi zaman P Olasilik degeri cok yüksek ciktigi icin ;

#coklu dogrusl regresyon uygulama
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


                            




