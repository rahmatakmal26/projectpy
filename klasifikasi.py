import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Memuat dataset gambar
dataset = []
labels = []

# Mengumpulkan data gambar dan label dari direktori dataset
for label in ['manggis_bagus', 'manggis_busuk']:
    path = f'Manggis/{label}'
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        img = cv2.resize(img, (32, 32))  # Mengubah ukuran gambar
        dataset.append(img.flatten())
        labels.append(label)

# Mengonversi label menjadi nilai numerik
le = LabelEncoder()
labels = le.fit_transform(labels)

# Memisahkan dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Mengatur jumlah tetangga terdekat
knn.fit(X_train, y_train)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk memprediksi gambar
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img_flat = img.flatten()
    prediction = knn.predict([img_flat])
    label = le.inverse_transform(prediction)[0]
    return label

# Template HTML
HTML_TEMPLATE = """
<!doctype html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Klasifikasi Gambar Manggis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f8ff;
            color: #333;
        }
        h1 {
            color: #4a4a4a;
            text-align: center;
            margin-bottom: 30px;
        }
        form {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        input[type="file"] {
            display: none;
        }
        .file-upload {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .file-upload:hover {
            background-color: #45a049;
        }
        input[type="submit"] {
            display: block;
            width: 100%;
            padding: 10px;
            background-color: #008CBA;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
            transition: background-color 0.3s;
        }
        input[type="submit"]:hover {
            background-color: #007B9A;
        }
        #preview {
            max-width: 100%;
            margin-top: 20px;
            border-radius: 8px;
            display: none;
        }
        .result {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .result h2 {
            color: #4a4a4a;
            margin-top: 0;
        }
    </style>
</head>
<body>
    <h1>Klasifikasi Gambar Manggis</h1>
    <form method="post" enctype="multipart/form-data">
        <label for="file-upload" class="file-upload">
            Pilih Gambar
        </label>
        <input id="file-upload" type="file" name="file" accept=".png,.jpg,.jpeg" required onchange="previewImage(event)">
        <img id="preview" src="#" alt="Preview gambar">
        <input type="submit" value="Klasifikasikan">
    </form>
    {% if result %}
        <div class="result">
            <h2>Hasil Klasifikasi:</h2>
            <p>Gambar tersebut adalah {{ result }}</p>
        </div>
    {% endif %}

    <script>
        function previewImage(event) {
            var reader = new FileReader();
            reader.onload = function(){
                var output = document.getElementById('preview');
                output.src = reader.result;
                output.style.display = 'block';
            };
            reader.readAsDataURL(event.target.files[0]);
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Tidak ada file yang diunggah'
        file = request.files['file']
        if file.filename == '':
            return 'Tidak ada file yang dipilih'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_image(filepath)
            os.remove(filepath)  # Hapus file setelah diklasifikasi
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True)