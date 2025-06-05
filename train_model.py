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

    # Load and filter training data
    df = pd.read_csv(log_file)
    df = df[(df['user_feedback'] == 'No') & (df['corrected_emotion'].notnull())]

    if df.empty:
        print("❌ No valid rows after filtering. Make sure you submitted feedback with corrected emotions.")
        return

    # Collect features and labels
    X, y = [], []
    for _, row in df.iterrows():
        audio_path = row['file_name']
        if os.path.exists(audio_path):
            try:
                features = extract_features(audio_path)
                X.append(features)
                y.append(row['corrected_emotion'])
            except Exception as e:
                print(f"⚠️ Failed to extract features from {audio_path}: {e}")
        else:
            print(f"⚠️ File not found: {audio_path}")

    if not X:
        print("❌ No valid feature data available for training.")
        return

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Save model
    joblib.dump(clf, model_file)
    print(f"✅ Model trained and saved to {model_file}")

if __name__ == "__main__":
    train_emotion_model()

