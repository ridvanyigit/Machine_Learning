import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Veriyi okuma
df = pd.read_csv('musteriler.csv')

# Veri setini inceleme
print(df.head())

# Gerekli sütunları seçme ve One-Hot Encoding işlemi
data = df[['Cinsiyet', 'Yas', 'Hacim', 'Maas']]
data = pd.get_dummies(data, columns=['Cinsiyet'], drop_first=True)

# K-Means modeli oluşturma
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(data)
print("Küme Merkezleri:")
print(kmeans.cluster_centers_)

# Elbow Method kullanarak küme sayısını belirleme
results = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123)
    kmeans.fit(data)
    results.append(kmeans.inertia_)

# Elbow Method görselleştirme
plt.plot(range(1, 11), results, marker='o')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('İnertia')
plt.title('Elbow Method: Küme Sayısı Belirleme')
plt.show()

