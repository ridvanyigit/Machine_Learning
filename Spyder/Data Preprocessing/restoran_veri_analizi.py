import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri setini yükleme
tips = sns.load_dataset("tips")

# Veri setinin incelenmesi
# ----------------------------------------
# Sütun başlıklarının incelenmesi
print("Sütun Başlıkları:")
print(tips.columns)

# Veri tiplerinin kontrolü
print("\nVeri Tipleri:")
print(tips.dtypes)

# Eksik verilerin incelenmesi
print("\nEksik Verilerin Sayısı:")
print(tips.isnull().sum())

# Örnek satırların gözden geçirilmesi
print("\nÖrnek Satırlar:")
print(tips.head())

# Temel istatistiklerin hesaplanması
# ----------------------------------------
tips_num = tips.select_dtypes(include=['float64', 'int64'])
print("\nOrtalama Değerler:")
print(tips_num.mean())

print("\nStandart Sapmalar:")
print(tips_num.std())

print("\nMinimum Değerler:")
print(tips_num.min())

print("\nMaksimum Değerler:")
print(tips_num.max())

print("\nMedyanlar:")
print(tips_num.median())

print("\nÇeyrekler Arası Aralık:")
print(tips_num.quantile(0.75) - tips_num.quantile(0.25))

# Veri setini görselleştirme
# ----------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(tips['total_bill'], bins=10, color='blue', alpha=0.7)
plt.title('Hesap Miktarının Dağılımı')
plt.xlabel('Hesap Miktarı')
plt.ylabel('Frekans')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(y=tips['total_bill'])
plt.title('Total Bill Kutu Grafiği')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(tips['total_bill'], kde=True)
plt.title('Hesap Miktarı Dağılım Grafiği')
plt.xlabel('Hesap Miktarı')
plt.show()

correlation_matrix = tips_num.corr()
print("\nKorelasyon Matrisi:")
print(correlation_matrix)

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Sütunlar Arasındaki Korelasyon')
plt.show()

# Eksik verilerin incelenmesi ve işlenmesi
# ----------------------------------------
print("\nEksik Değerlerin Sayısı:")
print(tips.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(tips.isnull(), cbar=False)
plt.title('Eksik Değerlerin Dağılımı')
plt.show()

ortalama_tips = tips.fillna(tips_num.mean())
print("\nEksik Değerlerin Doldurulmuş Hali:")
print(ortalama_tips)

# Veri setinin anlamlandırılması
# ----------------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Sütunlar Arasındaki Korelasyon')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.title('Total Bill ve Tip İlişkisi')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Gün ve Total Bill İlişkisi')
plt.xlabel('Gün')
plt.ylabel('Total Bill')
plt.show()

sns.pairplot(tips)
plt.show()

sns.boxplot(data=tips, orient="h")
plt.title("Box Plot")
plt.show()

sns.countplot(x='day', data=tips)
plt.title("Gunlere Gore Odenen Hesap Sayisi")
plt.show()

tips['total_per_person'] = tips['total_bill'] / tips['size']
print("\nÖzellikler Eklenmiş Hali:")
print(tips.head())

