from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd  # <--- PENTING: Jangan lupa import pandas
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ===============================
# LOAD MODEL & PREPROCESSOR
# ===============================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    # Pastikan nama file sesuai dengan yang ada di folder model kamu
    kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    cluster_mapping = joblib.load(os.path.join(MODEL_DIR, "cluster_mapping.joblib"))

    print("[INFO] Model, scaler, dan mapping berhasil dimuat.")
except Exception as e:
    print("[ERROR] Model gagal dimuat:", e)
    # Opsional: Bisa tambahkan exit() jika model gagal load agar server tidak jalan sia-sia

# ===============================
# API PREDIKSI
# ===============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # === 1. Ambil input ===
        # Kita simpan nilai asli ke variabel terpisah untuk analisa nanti
        val_tugas = float(data['rata_tugas']) # Tambah float untuk jaga-jaga input string
        val_uts = float(data['nilai_uts'])
        val_uas = float(data['nilai_uas'])

        input_df = pd.DataFrame([{
            'rata_tugas': val_tugas,
            'nilai_uts': val_uts,
            'nilai_uas': val_uas
        }])

        # === 2. Scaling ===
        X_scaled = scaler.transform(input_df)

        # === 3. Prediksi cluster ===
        # PERBAIKAN: Gunakan 'kmeans_model', bukan 'model'
        cluster_id = int(kmeans_model.predict(X_scaled)[0]) 
        
        # Ambil label dari mapping (pastikan keys mapping berupa integer)
        label = cluster_mapping[cluster_id]

        # === 4. Hitung jarak (Confidence) ===
        # PERBAIKAN: Gunakan 'kmeans_model', bukan 'model'
        distances = np.linalg.norm(
            kmeans_model.cluster_centers_ - X_scaled,
            axis=1
        )
        total_distance = distances.sum()
        
        # NOTE: Pastikan urutan index [0], [1], [2] ini sesuai dengan urutan cluster
        # di model kamu (mana yang rendah, sedang, tinggi).
        confidence = {
            'aman': round((1 - distances[2] / total_distance) * 100, 2),
            'pantau': round((1 - distances[1] / total_distance) * 100, 2),
            'bimbingan': round((1 - distances[0] / total_distance) * 100, 2)
        }

        # === 5. Tentukan status UI & Rekomendasi Detail ===
        recommendation_list = []
        status_ui = ''

        # A. Logika Dasar Berdasarkan Cluster
        if label == 'Potensi Tinggi':
            status_ui = 'Aman'
            recommendation_list.append("Kinerja akademik sangat baik.")
            
            # Cek jika ada nilai yang 'jomplang' walau masuk kategori aman
            if val_tugas < 80:
                recommendation_list.append("Namun, kedisiplinan pengumpulan tugas perlu ditingkatkan agar sempurna.")
            else:
                recommendation_list.append("Disarankan mengikuti program pengayaan atau kompetisi akademik.")

        elif label == 'Potensi Cukup':
            status_ui = 'Pantau'
            recommendation_list.append("Kinerja cukup baik namun belum konsisten.")
            
            # Analisa mendalam: Kenapa dia cuma 'Cukup'?
            if val_tugas < 70:
                recommendation_list.append("Perhatian Khusus: Nilai tugas harian rendah. Perlu lebih rajin mengerjakan latihan.")
            elif val_uts < 70 or val_uas < 70:
                recommendation_list.append("Perhatian Khusus: Pemahaman materi ujian kurang. Disarankan belajar kelompok atau review materi ulang.")
            else:
                recommendation_list.append("Tingkatkan fokus di kelas untuk mendongkrak nilai ke level atas.")

        else: # Potensi Rendah
            status_ui = 'Bimbingan'
            recommendation_list.append("Perlu intervensi segera.")
            
            # Analisa mendalam: Apa masalah utamanya?
            if val_tugas < 60 and (val_uts > 60 or val_uas > 60):
                recommendation_list.append("Siswa memiliki kemampuan (nilai ujian baik) namun kurang disiplin mengerjakan tugas.")
            elif (val_uts < 55 or val_uas < 55) and val_tugas > 70:
                recommendation_list.append("Siswa rajin tugas tapi kesulitan saat ujian. Perlu pendampingan belajar intensif.")
            else:
                recommendation_list.append("Wajib dijadwalkan sesi konseling akademik dan remedial menyeluruh.")

        # Gabungkan list menjadi satu kalimat string
        final_recommendation = " ".join(recommendation_list)

        # === 6. Response JSON ===
        return jsonify({
            'status': 'success',
            'prediction': {
                'cluster_id': cluster_id,
                'label': label,
                'status_ui': status_ui,
                'confidence_percent': confidence,
                'scores': {
                    'rata_tugas': val_tugas,
                    'nilai_uts': val_uts,
                    'nilai_uas': val_uas
                },
                'recommendation': final_recommendation
            }
        })

    except Exception as e:
        # Print error di terminal agar mudah debug
        print(f"[ERROR PREDICT]: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)