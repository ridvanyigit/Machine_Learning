import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Seaborn kütüphanesinden "tips" veri setini yükleyelim
tips = sns.load_dataset("tips")

# Veri setini görüntüleyelim
print(tips)


# ======================1.ADIM: Veri Setini Inceleme===========================
# Sütun Başlıklarının İncelenmesi:        Her bir sütunun adı ve neyi temsil ettiği hakkında bilgi alınır.
print(tips.columns)
# Veri Tiplerinin Kontrolü:               Her bir sütunun veri türü kontrol edilir (sayısal, kategorik, tarih gibi).
print(tips.dtypes)
# Eksik Verilerin İncelenmesi:            Veri setindeki eksik değerlerin bulunup bulunmadığı kontrol edilir.
print(tips.isnull().sum())
# Örnek Satırların Gözden Geçirilmesi:    Veri setindeki ilk birkaç satır incelenerek verinin genel yapısı hakkında fikir edinilir.
print(tips.head())
# =============================================================================


# =====================2. ADIM: Temel İstatistiklerin Hesaplanması=============
# Ortalama Değerlerin Hesaplanması:                Her sayısal sütun için ortalama değerler hesaplanır.
sayisal_sutunlar = tips.select_dtypes(include=['float64', 'int64'])
ortalama_degerler = sayisal_sutunlar.mean()
print(ortalama_degerler)
# Standart Sapmanın Hesaplanması:                  Sayısal sütunlar için standart sapma hesaplanır.
standart_sapmalar = sayisal_sutunlar.std()
print(standart_sapmalar)
# Minimum ve Maksimum Değerlerin Bulunması:        Her sayısal sütun için minimum ve maksimum değerler belirlenir.
min_degerler = sayisal_sutunlar.min()
max_degerler = sayisal_sutunlar.max()
print("Minimum Değerler:")
print(min_degerler)
print("\nMaksimum Değerler:")
print(max_degerler)
# Medyan ve Çeyrekler Arası Aralığın Hesaplanması: Medyan ve çeyrekler arası aralık gibi diğer istatistikler hesaplanır.
medyanlar = sayisal_sutunlar.median()
ceyrekler_arasi_aralik = sayisal_sutunlar.quantile(0.75) - sayisal_sutunlar.quantile(0.25)   # %50'lik agirligi ölcer, kutu grafikleri icin gereklidir.
print("Medyanlar:")
print(medyanlar)
print("\nÇeyrekler Arası Aralık:")
print(ceyrekler_arasi_aralik)
# =============================================================================



# ====================3.ADIM: Veri Setini Görselleştirme:======================
# Histogramlar: Veri setindeki dağılımları görselleştirmek için histogramlar kullanılır.
plt.hist(tips['total_bill'], bins=10, color='blue', alpha=0.7)
plt.title('Hesap Miktarının Dağılımı')
plt.xlabel('Hesap Miktarı')
plt.ylabel('Frekans')
plt.show()
# Kutu Grafikleri (Box Plots): Veri setindeki aykırı değerleri ve dağılımları görselleştirmek için kutu grafikleri kullanılır.
sns.boxplot(y=tips['total_bill'])    
# Dağılım Grafikleri: Veri setindeki ilişkileri ve dağılımları görselleştirmek için dağılım grafikleri kullanılır.
sns.histplot(tips['total_bill'], kde=True)
# Korelasyon Matrisleri: Veri setindeki sütunlar arasındaki ilişkileri görselleştirmek için korelasyon matrisleri kullanılır.
correlation_matrix = sayisal_sutunlar.corr()
print(correlation_matrix)
# =============================================================================



# ===================4. ADIM: Eksik Verilerin İncelenmesi ve İşlenmesi=========
# Eksik Verilerin Belirlenmesi: Veri setindeki eksik değerlerin sayısı ve dağılımı incelenir.
eksik_deger_sayisi = tips.isnull().sum()
print("Eksik Değerlerin Sayısı:")
print(eksik_deger_sayisi)

sns.heatmap(tips.isnull(), cbar=False)
plt.title('Eksik Değerlerin Dağılımı')
plt.show()
# Eksik Verilerin İşlenmesi: Eksik değerler, doldurma, silme veya başka bir yöntemle işlenir.
ortalama_tips = tips.fillna(sayisal_sutunlar.mean())
print(ortalama_tips)
# =============================================================================




# =================5. ADIM: Veri Setinin Anlamlandırılması=====================
# Sütunlar Arasındaki İlişkilerin İncelenmesi: Farklı sütunlar arasındaki ilişkiler ve etkileşimler incelenir.
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Sütunlar Arasındaki Korelasyon')
plt.show()          

plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips)    # -> Bir sayisal sütunun baska bir sayisal sütunla olan iliskisi (total_bill & tip)
plt.title('Total Bill ve Tip İlişkisi')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='day', y='total_bill', data=tips)      # -> Bir Kategorik sütun ile bir Sayisal sütun arasindaki iliski (day & total_bill)
plt.title('Gün ve Total Bill İlişkisi')
plt.xlabel('Gün')
plt.ylabel('Total Bill')
plt.show()
# Özelliklerin Analizi: Veri setindeki özelliklerin önemi ve etkisi değerlendirilir.

# Özellik Dağılımlarının İncelenmesi
sns.pairplot(tips)
plt.show()

# Özellikler Arasındaki İlişkilerin İncelenmesi
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

# Anormalliklerin ve Aykırı Değerlerin İncelenmesi
sns.boxplot(data=tips, orient="h")
plt.title("Box Plot")
plt.show()

# Kategorik Özelliklerin İncelenmesi
sns.countplot(x='day', data=tips)
plt.title("Gunlere Gore Odenen Hesap Sayisi")
plt.show() 

# Özellik Mühendisliği
tips['total_per_person'] = tips['total_bill'] / tips['size']
print(tips.head())
