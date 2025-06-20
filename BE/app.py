from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os, sqlite3
from werkzeug.utils import secure_filename
from flask_cors import CORS
from datetime import datetime
import cv2
from utils.auth import register_user, login_user  # pastikan file ini ada

# === Konfigurasi Flask ===
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Kelas Kategori ===
class_names = [
    "Fully Nutritional (Sehat)",
    "Kekurangan Kalium (K)",
    "Kekurangan Nitrogen (N)",
    "Kekurangan Fosfor (P)"
]

# === Load Model TFLite ===
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Fungsi Deteksi Apakah Gambar Daun Berdasarkan Warna ===
def is_leaf_image(image_path, green_threshold=0.15):
    img = cv2.imread(image_path)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = np.count_nonzero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    green_ratio = green_pixels / total_pixels
    return green_ratio >= green_threshold

# === Endpoint Registrasi ===
@app.route('/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if register_user(username, password):
        return jsonify({'message': 'Registrasi berhasil'}), 201
    return jsonify({'message': 'Username sudah terdaftar'}), 400

# === Endpoint Login ===
@app.route('/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user_id = login_user(username, password)
    if user_id:
        return jsonify({'message': 'Login berhasil', 'user_id': user_id}), 200
    return jsonify({'message': 'Username/password salah'}), 401

# === Endpoint Prediksi ===
@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.form.get('user_id')
    file = request.files.get('file')

    if not user_id or not file:
        return jsonify({'error': 'user_id dan file diperlukan'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Cek apakah gambar adalah daun
    if not is_leaf_image(filepath):
        return jsonify({'error': 'Gambar bukan daun. Harap unggah gambar daun selada yang jelas.'}), 400

    # Preprocessing
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    img_array = img_array.astype(np.float32)

    # Prediksi
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(output_data)]
    confidence = float(np.max(output_data) * 100)

    # Simpan hasil ke database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO riwayat (user_id, filename, prediction, timestamp) VALUES (?, ?, ?, ?)",
                   (user_id, filename, predicted_class, datetime.now()))
    conn.commit()
    conn.close()

    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence, 2),
        'filename': filename
    })

# === Endpoint Riwayat ===
@app.route('/riwayat/<int:user_id>', methods=['GET'])
def riwayat(user_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT filename, prediction, timestamp FROM riwayat WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    records = cursor.fetchall()
    conn.close()

    riwayat_list = []
    for filename, prediction, timestamp in records:
        riwayat_list.append({
            'filename': filename,
            'prediction': prediction,
            'timestamp': timestamp
        })

    return jsonify({'riwayat': riwayat_list})

# === Endpoint Gambar Upload ===
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# === Jalankan Server ===
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
