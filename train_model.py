import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from features import extract_features

def train_emotion_model(log_file="results_log.csv", model_file="emotion_model.pkl"):
    if not os.path.exists(log_file):
        print(f"❌ No training data found at {log_file}")
        return

    df = pd.read_csv(log_file)

    # Only use rows where the user provided corrected labels
    df = df[(df['user_feedback'] == 'No') & (df['corrected_emotion'].notnull())]

    print(f"🔍 Found {len(df)} rows with corrected emotion feedback.")

    X, y = [], []

    for _, row in df.iterrows():
        audio_path = row['file_name']

        if not os.path.exists(audio_path):
            print(f"⚠️ File not found: {audio_path}")
            continue

        try:
            features = extract_features(audio_path)
            X.append(features)
            y.append(row['corrected_emotion'])
        except Exception as e:
            print(f"❌ Failed to extract features from {audio_path}: {e}")

    if not X:
        print("❌ No valid data for training.")
        return

    X = np.array(X)
    y = np.array(y)

    clf = RandomForestClassifier()
    clf.fit(X, y)
    joblib.dump(clf, model_file)
    print(f"✅ Model trained and saved as {model_file}")

if __name__ == "__main__":
    train_emotion_model()
