import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import os

# === PATH CONFIG ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# === LOAD DATASET KAGGLE ===
dataset_path = os.path.join(DATASET_DIR, "Student_Performance.csv")
df = pd.read_csv(dataset_path)

# === TRANSFORMASI FITUR (KUNCI SKRIPSI) ===
# Nilai UTS dari Previous Scores
df['nilai_uts'] = df['Previous Scores']

# Nilai Tugas dari Sample Question Papers Practiced
max_task = df['Sample Question Papers Practiced'].max()
df['rata_tugas'] = (df['Sample Question Papers Practiced'] / max_task) * 100

# Nilai UAS dari Performance Index
df['nilai_uas'] = df['Performance Index']

features = ['rata_tugas', 'nilai_uts', 'nilai_uas']
X = df[features]

# === SCALING ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === ELBOW METHOD ===
inertia = []
K_range = range(1, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Metode Elbow')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig(os.path.join(IMAGE_DIR, "elbow.png"))
plt.close()

# === TRAINING K-MEANS ===
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# === EVALUASI ===
score = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score: {score:.4f}")

# === LABELING CLUSTER ===
df['cluster'] = kmeans.labels_

cluster_means = df.groupby('cluster')[features].mean().mean(axis=1).sort_values()

cluster_mapping = {
    cluster_means.index[0]: 'Potensi Kurang',
    cluster_means.index[1]: 'Potensi Cukup',
    cluster_means.index[2]: 'Potensi Tinggi'
}

df['label'] = df['cluster'].map(cluster_mapping)

# === SAVE MODEL ===
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(cluster_mapping, os.path.join(MODEL_DIR, "cluster_mapping.joblib"))

print("Model training selesai dan siap digunakan.")
