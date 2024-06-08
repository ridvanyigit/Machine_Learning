import csv
from apyori import apriori

# İlişki kayıtlarını düzenli bir şekilde yazdırma fonksiyonu
def print_rules(rules):
    for i, rule in enumerate(rules, start=1):
        items_base = ', '.join(list(rule.items)[0])
        items_add = ', '.join(list(rule.items)[1]) if len(rule.items) > 1 else ""
        support = round(rule.support, 4)
        confidence = round(rule.ordered_statistics[0].confidence, 4)
        lift = round(rule.ordered_statistics[0].lift, 4)

        print(f"{i}. {{{items_base}}} => {{{items_add}}}")
        print(f"   - Destek (Support): {support}")
        print(f"   - Güven (Confidence): {confidence}")
        print(f"   - Lift: {lift}\n")

# Veri setini yükle
with open('sepet.csv', 'r') as file:
    reader = csv.reader(file)
    transactions = list(reader)

# Apriori algoritması ile kuralları bul
rules = apriori(transactions, min_support=0.05, min_confidence=0.2, min_lift=1, min_length=2)

# İlişki kayıtlarını yazdır
print_rules(rules)

