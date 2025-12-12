# Gunakan image Python yang ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements dulu (biar caching docker optimal)
COPY requirements.txt .

# Install dependencies
# --no-cache-dir biar image-nya kecil
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode (app.py, model.pkl, scaler.pkl, dll)
COPY . .

# Expose port flask (Default 5000)
EXPOSE 5000

# Perintah untuk menjalankan aplikasi
# Opsi 1: Pakai Gunicorn (Disarankan untuk Production/Render/Railway)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

# Opsi 2: Pakai python biasa (Kalau di app.py sudah ada app.run)
# CMD ["python", "app.py"]