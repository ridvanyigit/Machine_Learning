import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Veriyi okuma
df = pd.read_csv('musteriler.csv')

# Veri setini inceleme
print(df.head())

# Gerekli sütunları seçme ve One-Hot Encoding işlemi
data = df[['Cinsiyet', 'Yas', 'Hacim', 'Maas']]
data = pd.get_dummies(data, columns=['Cinsiyet'], drop_first=True)

# Dendrogram oluşturma
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Müşteriler')
plt.ylabel('Euclidean Mesafesi')
plt.show()

# Kümeleme modeli oluşturma ve veriyi fit etme
ac = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
y_pred = ac.fit_predict(data)
print(y_pred)

# Sonuçları görselleştirme
plt.figure(figsize=(10, 7))

# Her bir kümeyi farklı bir renkle çizme
plt.scatter(data.iloc[y_pred == 0, 1], data.iloc[y_pred == 0, 2], s=100, c='red', label='Cluster 1')
plt.scatter(data.iloc[y_pred == 1, 1], data.iloc[y_pred == 1, 2], s=100, c='blue', label='Cluster 2')
plt.scatter(data.iloc[y_pred == 2, 1], data.iloc[y_pred == 2, 2], s=100, c='green', label='Cluster 3')
plt.scatter(data.iloc[y_pred == 3, 1], data.iloc[y_pred == 3, 2], s=100, c='orange', label='Cluster 4')

# Grafik ayarları
plt.title('Agglomerative Clustering')
plt.xlabel('Yas')
plt.ylabel('Hacim')
plt.legend()
plt.show()
