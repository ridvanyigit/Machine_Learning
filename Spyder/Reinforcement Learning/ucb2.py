import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Veri setini okuma
veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

# UCB algoritması
def ucb(veriler, N, d):
    oduller = [0] * d # Her kola ait toplam ödül
    tiklamalar = [0] * d # Her kola ait toplam tıklama sayısı
    toplam = 0 # Toplam ödül
    secilenler = [] # Her adımda seçilen kol

    for n in range(1, N):
        secilen_ad = 0 # Seçilen kol
        max_ucb = 0 # Maksimum UCB değeri
        
        for i in range(0, d):
            if tiklamalar[i] > 0:
                ortalama = oduller[i] / tiklamalar[i]
                delta = math.sqrt(3/2 * math.log(n) / tiklamalar[i])
                ucb_degeri = ortalama + delta
            else:
                # Henüz seçilmemiş kol varsa, onları seç
                ucb_degeri = N * 10
                
            if max_ucb < ucb_degeri:
                max_ucb = ucb_degeri
                secilen_ad = i
                
        secilenler.append(secilen_ad)
        tiklamalar[secilen_ad] += 1
        odul = veriler.values[n, secilen_ad] # Verideki n. satırdaki değer
        oduller[secilen_ad] += odul
        toplam += odul

    return toplam, secilenler, oduller, tiklamalar

# Simülasyon parametreleri
N = 10000 # Toplam adım sayısı
d = 10 # Toplam kol sayısı

# UCB algoritmasını çalıştırma
toplam_odul, secilen_kollar, tahmini_oduller, kol_secimleri = ucb(veriler, N, d)

# Sonuçları görselleştirme
plt.hist(secilen_kollar)
plt.title('UCB Algoritması ile Kolların Seçilme Sıklığı')
plt.xlabel('Kol Indeksi')
plt.ylabel('Seçilme Sayısı')
plt.show()

# Optimal toplam ödülü hesapla
optimal_toplam_odul = np.sum(veriler.max(axis=1))

# İdeal stratejinin performansını gösteren yüzdeyi hesapla
optimal_oran = (toplam_odul / optimal_toplam_odul) * 100
print("Optimal Stratejiye Göre Başarı Oranı: {:.2f}%".format(optimal_oran))

# Kol Seçimleri ve Tahmini Ödüller
print("Kol Seçimleri: ", kol_secimleri)
print("Tahmini Ödüller: ", [round(odul / tiklama, 2) if tiklama > 0 else 0 for odul, tiklama in zip(tahmini_oduller, kol_secimleri)])

# En çok seçilen kolu bulma
en_cok_secilen_kol = np.argmax(kol_secimleri)
print(f"En çok seçilen kol: {en_cok_secilen_kol}")

