from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os, sqlite3
from werkzeug.utils import secure_filename
from flask_cors import CORS
from datetime import datetime
from utils.auth import register_user, login_user  # kamu sudah punya ini

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Load TFLite model ===
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Label kelas ===
class_names = [
    "Fully Nutritional (Sehat)",
    "Kekurangan Kalium (K)",
    "Kekurangan Nitrogen (N)",
    "Kekurangan Fosfor (P)"
]

# === REGISTER ===
@app.route('/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if register_user(username, password):
        return jsonify({'message': 'Registrasi berhasil'}), 201
    return jsonify({'message': 'Username sudah terdaftar'}), 400

# === LOGIN ===
@app.route('/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user_id = login_user(username, password)
    if user_id:
        return jsonify({'message': 'Login berhasil', 'user_id': user_id}), 200
    return jsonify({'message': 'Username/password salah'}), 401

# === PREDIKSI ===
@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.form.get('user_id')
    file = request.files.get('file')

    if not user_id or not file:
        return jsonify({'error': 'user_id dan file diperlukan'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocessing gambar
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    img_array = img_array.astype(np.float32)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(output_data)]
    confidence = float(np.max(output_data) * 100)

    # Simpan ke database
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

# === RIWAYAT ===
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
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# === Start server ===
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=5001, debug=True)
