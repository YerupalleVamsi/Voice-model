import librosa
import numpy as np

def extract_features(file):
    y, sr = librosa.load(file, duration=3, offset=0.5)

    # Pitch extraction (mean of max pitch values per frame)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)
    pitch_mean = np.mean(pitch_values) if pitch_values else 0

    # Intensity (RMS energy)
    rms = librosa.feature.rms(y=y)[0]
    intensity_mean = np.mean(rms)

    # MFCCs mean
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = mfccs.mean(axis=1)

    features = np.hstack([pitch_mean, intensity_mean, mfccs_mean])
    return features
