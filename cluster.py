import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

path = "clustet.csv"
df = pd.read_csv(path)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

K_range = range(2, 13)
wcss = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    wcss.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    
    print(f"K={k}: WCSS={kmeans.inertia_:.0f}, Silhouette={sil:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Количество кластеров K', fontsize=12)
axes[0].set_ylabel('WCSS (сумма квадратов внутри кластеров)', fontsize=12)
axes[0].set_title('Метод локтя', fontsize=14)
axes[0].grid(True, alpha=0.3)
