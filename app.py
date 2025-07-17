import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_HEIGHT = 288
IMG_WIDTH = 720

MODEL_PATH = 'music_genre_classifier.h5'
model = load_model(MODEL_PATH)

CLASS_NAMES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def preprocess_audio(audio_path, spectrogram_path):
    try:
        y, sr = librosa.load(audio_path)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return False

@app.route('/', methods=['GET'])
def index():
    prediction = request.args.get('prediction', None)
    return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(audio_path)

        spectrogram_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_spectrogram.png')

        if not preprocess_audio(audio_path, spectrogram_path):
            return "Error processing audio file.", 500

        img = tf.keras.utils.load_img(
            spectrogram_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = CLASS_NAMES[np.argmax(score)]
        
        return redirect(url_for('index', prediction=predicted_class))

if __name__ == '__main__':
    app.run(debug=True)
