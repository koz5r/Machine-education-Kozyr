import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import time

X, _ = make_blobs(n_samples=150, centers=3, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42, max_iter=10)
kmeans.fit(X)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6)
initial_centers = kmeans.cluster_centers_.copy()
plt.scatter(initial_centers[:, 0], initial_centers[:, 1],
            c='red', marker='X', s=200, linewidths=3)
plt.title('Начальные центры')
plt.xlabel('Пр 1')
plt.ylabel('Пр 2')

plt.subplot(2, 2, 2)
labels_1 = kmeans.fit_predict(X)
centers_1 = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=labels_1, cmap='viridis', alpha=0.6)
plt.scatter(centers_1[:, 0], centers_1[:, 1],
            c='red', marker='X', s=200, linewidths=3)
plt.title('После первой итерации')

plt.subplot(2, 2, 3)
kmeans.n_init = 1
kmeans.max_iter = 2
kmeans.fit(X)
labels_2 = kmeans.labels_
centers_2 = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=labels_2, cmap='viridis', alpha=0.6)
plt.scatter(centers_2[:, 0], centers_2[:, 1],
            c='red', marker='X', s=200, linewidths=3)
plt.title('После второй итерации')

# Финальный результат
plt.subplot(2, 2, 4)
kmeans.max_iter = 100
kmeans.fit(X)
final_labels = kmeans.labels_
final_centers = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', alpha=0.6)
plt.scatter(final_centers[:, 0], final_centers[:, 1],
            c='red', marker='X', s=200, linewidths=3)
plt.title(f'Финальный результат после {kmeans.n_iter_} итераций')

plt.tight_layout()
plt.show()

# Анимация
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 6))
kmeans_anim = KMeans(n_clusters=3, random_state=42, max_iter=1)


def update(frame):
    ax.clear()
    if frame == 0:
        kmeans_anim.fit(X)
        centers = kmeans_anim.cluster_centers_
        labels = kmeans_anim.predict(X)
    else:
        kmeans_anim.max_iter = 1
        kmeans_anim.fit(X)
        centers = kmeans_anim.cluster_centers_
        labels = kmeans_anim.labels_

    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1],
               c='red', marker='X', s=200, linewidths=3, edgecolors='white')
    ax.set_title(f'Итерация {frame + 1}')
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')

    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        for point in cluster_points[:30]:
            ax.plot([point[0], center[0]], [point[1], center[1]],
                    'gray', alpha=0.3, linewidth=0.5)

    return scatter


anim = FuncAnimation(fig, update, frames=5, repeat=False, interval=1500)
plt.tight_layout()
plt.show()