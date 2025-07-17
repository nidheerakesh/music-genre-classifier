import os
import pathlib
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

DATASET_PATH = "data/genres"

SPECTOGRAM_PATH = "data/spectrograms"

def create_spectrograms(dataset_path,spectogram_path):
    print("Starting preprocessing...")
    dataset_path = pathlib.Path(dataset_path)
    spectogram_path = pathlib.Path(spectogram_path)

    spectogram_path.mkdir(parents=True, exist_ok=True)

    for genre in dataset_path.iterdir():
        if genre.is_dir():
            print(f"Processing genre: {genre.name}")
            
            output_genre_dir= spectogram_path / genre.name
            output_genre_dir.mkdir(parents=True, exist_ok=True)

            for audio_file in genre.glob("*.au"):
                try:
                    output_file_path = output_genre_dir / f"{audio_file.stem}.png"
                    if output_file_path.exists():
                        print(f"Spectrogram for {audio_file.name} already exists, skipping.")
                        continue
                    
                    y, sr = librosa.load(audio_file, sr=None)
                    S = librosa.feature.melspectrogram(y=y, sr=sr)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    fig,ax = plt.subplots(figsize=(10, 4))

                    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
                    
                    ax.axis('off')
                    fig.tight_layout(pad=0)

                    fig.savefig(output_file_path, bbox_inches='tight', pad_inches=0 )
                    plt.close(fig)

                except Exception as e:
                    print(f"Error processing {audio_file.name}: {e}")
                    
    print("Preprocessing completed.")
if __name__ == "__main__":
    create_spectrograms(DATASET_PATH, SPECTOGRAM_PATH)
    print("Spectrograms created successfully.")
