import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import os

# === KONFIGURASI PATH (Relative Path agar aman dipindah-pindah) ===
# File ini ada di folder /training, jadi kita perlu naik satu level ke root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.dirname(CURRENT_DIR) # Naik ke D:.

DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
IMAGE_DIR = os.path.join(ROOT_DIR, "images") # Folder baru untuk simpan grafik skripsi

# Buat folder output jika belum ada
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print(f"[*] Root Directory: {ROOT_DIR}")

# === 1. Load Dataset ===
dataset_path = os.path.join(DATASET_DIR, "dataset_siswa.csv")
print(f"[*] Membaca dataset dari: {dataset_path}")
df = pd.read_csv(dataset_path)

features = ['rata_tugas', 'nilai_uts', 'nilai_uas']
X = df[features]

# === 2. Preprocessing (Scaling) ===
# Penting: K-Means menghitung jarak (Euclidean), jadi skala data harus sama
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# TAMBAHAN SKRIPSI: GRAFIK ELBOW (Mencari K Optimal)
# ==============================================================================
print("[*] Menghasilkan Grafik Elbow untuk Evaluasi K...")
inertia = []
K_range = range(1, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Metode Elbow untuk Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia (Total Dalam Cluster)')
plt.grid(True)
elbow_path = os.path.join(IMAGE_DIR, "grafik_elbow.png")
plt.savefig(elbow_path)
print(f"    -> Grafik Elbow disimpan di: {elbow_path}")
plt.close() # Tutup plot agar tidak menumpuk memori

# ==============================================================================
# === 3. Training Model K-Means (Utama) ===
# ==============================================================================
# Disini algoritma K-Means bekerja mencari titik tengah (centroid)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
kmeans.fit(X_scaled) 

# === 4. Evaluasi ===
score = silhouette_score(X_scaled, kmeans.labels_)
print(f"\n[*] Model berhasil dilatih dengan K={k_optimal}!")
print(f"[*] Silhouette Score: {score:.4f} (Indikator kualitas cluster)")

# === 5. Mapping Cluster & Visualisasi 3D ===
df['cluster'] = kmeans.labels_

# Hitung rata-rata tiap cluster untuk menentukan label (Kurang/Cukup/Tinggi)
cluster_means = df.groupby('cluster')[features].mean().mean(axis=1).sort_values()

# Mapping logis: Index rata-rata terendah = Kurang, dst.
cluster_mapping = {
    cluster_means.index[0]: 'Potensi Kurang',
    cluster_means.index[1]: 'Potensi Cukup',
    cluster_means.index[2]: 'Potensi Tinggi'
}

print("\n[*] Mapping Label Cluster:")
for k, v in cluster_mapping.items():
    print(f"    Cluster {k} -> {v}")

# Tambahkan label deskriptif ke dataframe untuk visualisasi
df['label'] = df['cluster'].map(cluster_mapping)

# ==============================================================================
# TAMBAHAN SKRIPSI: SCATTER PLOT 3D
# ==============================================================================
print("\n[*] Membuat Visualisasi 3D Cluster...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Warna untuk tiap label
colors = {'Potensi Kurang': 'red', 'Potensi Cukup': 'yellow', 'Potensi Tinggi': 'green'}

for label, color in colors.items():
    subset = df[df['label'] == label]
    ax.scatter(subset['rata_tugas'], subset['nilai_uts'], subset['nilai_uas'], 
               c=color, label=label, s=50, alpha=0.6)

ax.set_xlabel('Rata Tugas')
ax.set_ylabel('Nilai UTS')
ax.set_zlabel('Nilai UAS')
ax.set_title('Visualisasi Clustering Potensi Siswa (3D)')
ax.legend()

scatter_path = os.path.join(IMAGE_DIR, "grafik_cluster_3d.png")
plt.savefig(scatter_path)
print(f"    -> Grafik 3D disimpan di: {scatter_path}")
plt.close()

# === 6. Save Model ===
joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans_model.joblib'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
joblib.dump(cluster_mapping, os.path.join(MODEL_DIR, 'cluster_mapping.joblib'))

print("\n[*] SELESAI. Model & Grafik siap digunakan untuk Bab 4 Skripsi.")