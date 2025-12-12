import numpy as np
import pandas as pd

np.random.seed(42)

# --- 1. Jumlah data di masing-masing cluster ---
n_good = 200       # siswa performa bagus
n_medium = 180     # siswa performa sedang
n_low = 170         # siswa performa rendah

# --- 2. Generate cluster "good" ---
good_rata_tugas = np.random.normal(85, 5, n_good)
good_uts = np.random.normal(87, 5, n_good)
good_uas = np.random.normal(89, 5, n_good)

# --- 3. Generate cluster "medium" ---
medium_rata_tugas = np.random.normal(72, 6, n_medium)
medium_uts = np.random.normal(70, 7, n_medium)
medium_uas = np.random.normal(73, 6, n_medium)

# --- 4. Generate cluster "low" ---
low_rata_tugas = np.random.normal(58, 7, n_low)
low_uts = np.random.normal(55, 7, n_low)
low_uas = np.random.normal(57, 8, n_low)

# --- 5. Gabungkan ke dataframe ---
rata_tugas = np.concatenate([good_rata_tugas, medium_rata_tugas, low_rata_tugas])
uts = np.concatenate([good_uts, medium_uts, low_uts])
uas = np.concatenate([good_uas, medium_uas, low_uas])

# --- 6. Pastikan nilai dalam range 0–100 ---
def clamp(x):
    return np.clip(x, 0, 100)

df = pd.DataFrame({
    "rata_tugas": clamp(rata_tugas),
    "nilai_uts": clamp(uts),
    "nilai_uas": clamp(uas)
})

# --- 7. Acak urutan biar dataset lebih natural ---
df = df.sample(frac=1).reset_index(drop=True)

# --- 8. Save ke CSV ---
df.to_csv("dataset_siswa.csv", index=False)

df.head()
