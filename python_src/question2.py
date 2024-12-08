# Adımları Python'da uygulamaya başlayalım
# Gerekli kütüphaneleri çağıralım
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["OMP_NUM_THREADS"] = "1"


# Veri setini yükleyelim
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
columns = [
    "Class label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]
wine_data = pd.read_csv(url, header=None, names=columns)

# Basic information
print("\nDataset Information:")
print(wine_data.info())

# Check for missing values
print("\nMissing Values:")
print(wine_data.isnull().sum())

# Özellikler ve hedef değişkeni ayıralım
X = wine_data.drop("Class label", axis=1)  # Sadece özellikleri alıyoruz

# Veriyi standartlaştıralım
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Farklı k değerleri için k-means uygulayıp Silhouette Skorunu kontrol edelim
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Silhouette Skorunu görselleştirelim
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker="o")
plt.title("Silhouette Score for Different k Values")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

# En iyi k değerine karar verelim ve k-means uygulayalım
best_k = k_values[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_scaled)

# Sonuçları DataFrame'e ekleyelim
wine_data["Cluster"] = kmeans.labels_

# Grupların ortalamalarını inceleyelim
group_means = wine_data.groupby("Cluster").mean()
print(group_means)

# Cluster dağılımını görselleştirelim
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=X_scaled[:, 0], y=X_scaled[:, 1], hue=kmeans.labels_, palette="viridis", s=50
)
plt.title("Cluster Visualization")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.show()
