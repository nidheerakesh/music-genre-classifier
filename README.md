AI Music Genre Classifier

OVERVIEW
This project is a web application that uses a deep learning model to classify the genre of a music track from an audio file. It can identify 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

TECHNOLOGIES

Backend: Python, Flask

Machine Learning: TensorFlow, Keras

Audio & Data Processing: Librosa, NumPy

Data Visualization: Matplotlib

Dataset: GTZAN Genre Collection

SETUP AND INSTALLATION

Clone the Repository:
git clone https://github.com/nidheerakesh/music-genre-classifier.git
cd music-genre-classifier

Create and Activate Virtual Environment:
python -m venv venv
source venv/bin/activate  (On macOS/Linux)
venv\Scripts\activate    (On Windows)

Install Dependencies:
pip install -r requirements.txt

Download the Dataset:
Download the GTZAN Genre Collection dataset (from Kaggle or elsewhere). Unzip it and place the 'genres' folder at this path: data/genres/.

HOW TO USE

Preprocess the Data:
Run this command to turn the audio files into images.
python preprocess.py

Train the Model:
Run this command to train the AI model. This will create the music_genre_classifier.h5 file.
python train.py

Run the Application:
Run this command to start the web server.
python app.py
Then, open your web browser and go to http://127.0.0.1:5000/.
