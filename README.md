# 🎶 AI Music Genre Classifier

## 🚀 Overview

This project is a complete, end-to-end web application that uses a deep learning model to classify the **genre of a music track** from an audio file. It can identify **10 genres**:  
🎸 *blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.*

The core of the project is a **Convolutional Neural Network (CNN)** trained on spectrograms generated from the **GTZAN dataset**. The model is served via a **Flask API** with a clean user interface for uploading audio files and viewing predictions.

---

## ✨ Features

- 🎧 **Audio Preprocessing:** Converts raw `.au` audio files into Mel spectrogram images  
- 🧠 **Deep Learning Model:** Robust CNN architecture built with TensorFlow + Keras  
- 🌐 **Web Interface:** Simple and intuitive UI for audio upload  
- ⚡ **Real-time Prediction:** Instant genre classification on uploaded audio  

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask  
- **Machine Learning:** TensorFlow, Keras  
- **Audio Processing:** Librosa, NumPy  
- **Visualization:** Matplotlib  
- **Dataset:** GTZAN Genre Collection  

---

## 🔧 Setup & Installation

### 1. Clone the Repository
git clone https://github.com/nidheerakesh/music-genre-classifier.git
cd music-genre-classifier
2. Create and Activate a Virtual Environment
bash
Copy
Edit
# Create virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
💡 You can regenerate this file using:
pip freeze > requirements.txt

4. Download the Dataset
Download the GTZAN Genre Collection from Kaggle or other sources

Place the extracted folder in the following location:
data/genres/

🏃‍♀️ How to Use
1. Preprocess the Audio Data
python preprocess.py
Converts audio files to spectrograms and stores them in:


data/spectrograms/
2. Train the Model
python train.py
Trains a CNN on the spectrograms

Saves model to: music_genre_classifier.h5

3. Run the Web App
python app.py
Open your browser and go to:
http://127.0.0.1:5000/

📁 Project Structure
music_genre_classifier/
├── app.py                     # Flask web application
├── preprocess.py              # Converts audio to spectrograms
├── train.py                   # Trains CNN model
├── music_genre_classifier.h5  # Trained model (output)
├── requirements.txt           # Dependencies
├── templates/
│   └── index.html             # Frontend UI
├── data/
│   ├── genres/                # Raw GTZAN audio files (download separately)
│   └── spectrograms/          # Generated spectrogram images
├── uploads/                   # Temporary uploaded files
├── .gitignore
└── README.md
