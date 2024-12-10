import numpy as np
import librosa

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return np.hstack([np.mean(mfccs.T, axis=0), np.mean(chroma.T, axis=0), np.mean(mel.T, axis=0), np.mean(contrast.T, axis=0), np.mean(tonnetz.T, axis=0)])
