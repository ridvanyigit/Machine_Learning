import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('veriler.csv')
#print(veriler)


Yas = veriler.iloc[:,1:4].values                                               
#print(Yas)



#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
#print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
#print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
#print(ulke)





cinsiyet = veriler.iloc[:,-1].values
#print(cinsiyet)

#encoder: Kategorik -> Numeric
c = veriler.iloc[:,-1:].values
#print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
#print(c)
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
#print(c)




#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
#print(sonuc)
sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
#print(sonuc2)
# Bu kod satirinda Kukla Degisken Tuzagindan (Dummy Variable) kacinilmistir ve bir tane sütun isleme alinmistir.
sonuc3 = pd.DataFrame(data=c[:,:1], index = range(22), columns = ['cinsiyet'])   
#print(sonuc3)


#dataframe birlestirme islemi
s= pd.concat([sonuc,sonuc2], axis=1)
#print(s)
s2= pd.concat([s,sonuc3], axis=1)
#print(s2)



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)




#coklu dogrusl regresyon uygulama
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
#print(boy)

sol = s2.iloc[:,:3]            
sag = s2.iloc[:,4:]      

veri = pd.concat([sol,sag],axis=1)       

x_train, x_test, y_train, y_test = train_test_split(veri,boy,test_size=0.33,random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test) 


             #Basari modellin basarisini ölcüyoruz. Olasilik degeri (p_degeri) ölcümü
             
             
#Backward Elimination ( Geriye Dogru Eleme )
import statsmodels.api as sm       

X = np.append(arr= np.ones((22,1)).astype(int), values= veri, axis=1)      

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog= boy, exog=X_l)                 
r = r_ols.fit()
print(r.summary())              


#p degeri en yüksek olan 4.index'teki sütunu veri setinden sildik
X_l = veri.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog= boy, exog=X_l)                 
r = r_ols.fit()
print(r.summary())































