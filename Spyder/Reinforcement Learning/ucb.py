import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Veri setini okuma
data = pd.read_csv('Ads_CTR_Optimisation.csv')

# Veri setini kontrol etme
print("Veri seti boyutları:", data.shape)
print("Veri setinin ilk birkaç satırı:\n", data.head())

class UCBAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        total_counts = np.sum(self.counts)
        if total_counts < self.n_arms:
            return int(total_counts)  # Her kolu en az bir kez çekmek için
        ucb_values = self.values + np.sqrt((2 * np.log(total_counts)) / (self.counts + 1e-5))
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

# Simülasyon parametreleri
n_arms = data.shape[1]
n_simulations = data.shape[0]

# UCB ajanını oluşturma
agent = UCBAgent(n_arms)

# Simülasyonu yürütme
rewards = np.zeros(n_simulations)
chosen_arms = np.zeros(n_simulations)

for t in range(n_simulations):
    chosen_arm = agent.select_arm()
    # `chosen_arm`'ın geçerli bir indeks olduğundan emin olun
    if chosen_arm < 0 or chosen_arm >= n_arms:
        raise ValueError(f"Geçersiz kol seçimi: {chosen_arm}")
    
    reward = data.values[t, chosen_arm]  # Burada indekslemenin doğru olduğundan emin olun
    agent.update(chosen_arm, reward)
    rewards[t] = reward
    chosen_arms[t] = chosen_arm

# Sonuçları görselleştirme
cumulative_rewards = np.cumsum(rewards)
average_rewards = cumulative_rewards / (np.arange(n_simulations) + 1)

plt.figure(figsize=(12, 6))
plt.plot(average_rewards)
plt.xlabel("Simülasyon")
plt.ylabel("Ortalama Ödül")
plt.title("UCB Algoritması ile Ortalama Ödül")
plt.show()

# Toplam ödülü hesapla
total_reward = np.sum(rewards)
print("Toplam Ödül:", total_reward)

# Optimal toplam ödülü hesapla
optimal_total_reward = np.sum(data.values[np.arange(n_simulations), np.argmax(data.values, axis=1)])
print("Optimal Toplam Ödül:", optimal_total_reward)

# İdeal stratejinin performansını gösteren yüzdeyi hesapla
optimal_percentage = (total_reward / optimal_total_reward) * 100
print("Optimal Stratejiye Göre Başarı Oranı: {:.2f}%".format(optimal_percentage))

# Grafikteki ortalama ödülün zamanla nasıl değiştiğini gösterme
plt.figure(figsize=(12, 6))
plt.plot(average_rewards)
plt.xlabel("Simülasyon")
plt.ylabel("Ortalama Ödül")
plt.title("UCB Algoritması ile Ortalama Ödül")
plt.show()

print("Kol Seçimleri: ", agent.counts)
print("Tahmini Ödüller: ", agent.values)

# En çok seçilen kolu bulma
most_selected_arm = np.argmax(agent.counts)
print(f"En çok seçilen kol: {most_selected_arm}")

