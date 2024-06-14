#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:46:11 2024

@author: ridvanyigit
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# Veri setini yükleyin
data = pd.read_csv('Ads_CTR_Optimisation.csv')

# Thompson Sampling parametreleri
N = data.shape[0]  # Toplam kullanıcı sayısı
d = data.shape[1]  # Toplam reklam sayısı

# Başarı ve başarısızlık sayıları
successes = [0] * d
failures = [0] * d
total_reward = 0

# Thompson Sampling
ads_selected = []

for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(successes[i] + 1, failures[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = data.values[n, ad]
    if reward == 1:
        successes[ad] += 1
    else:
        failures[ad] += 1
    total_reward += reward

# Seçilen reklamların görselleştirilmesi
plt.hist(ads_selected, bins=np.arange(d+1)-0.5, edgecolor='black')
plt.title('Thompson Sampling ile Seçilen Reklamlar')
plt.xlabel('Reklamlar')
plt.ylabel('Seçim Sayısı')
plt.xticks(range(d))
plt.show()

print(f"Toplam ödül: {total_reward}")

