from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model yang sudah dilatih
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    cluster_mapping = joblib.load(os.path.join(MODEL_DIR, "cluster_mapping.joblib"))
    print("Model loaded successfully.")
except:
    print("Model not found. Jalankan train_model.py dulu!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Ambil data dari request body
        # Pastikan urutan kolom SAMA PERSIS dengan saat training
        input_data = [
            data['rata_tugas'],
            data['nilai_uts'],
            data['nilai_uas']
        ]
        
        # Ubah ke bentuk numpy array 2D
        features = np.array([input_data])
        
        # Lakukan scaling (Wajib!)
        features_scaled = scaler.transform(features)
        
        # Prediksi cluster
        prediction_cluster = model.predict(features_scaled)[0]
        
        # Map cluster ID ke Label manusia (Kurang/Cukup/Tinggi)
        prediction_label = cluster_mapping[prediction_cluster]
        
        response = {
            'status': 'success',
            'input_data': data,
            'prediction': {
                'cluster_id': int(prediction_cluster),
                'potensi': prediction_label
            }
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)