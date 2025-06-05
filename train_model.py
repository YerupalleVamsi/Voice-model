import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from features import extract_features

def train_emotion_model(log_file="results_log.csv", model_file="emotion_model.pkl"):
    if not os.path.exists(log_file):
        print(f"No training data found at {log_file}")
        return

    df = pd.read_csv(log_file)
    df = df[df['user_feedback'] == 'No']
    df = df[df['corrected_emotion'].notnull()]

    X, y = [], []

    for _, row in df.iterrows():
        if os.path.exists(row['file_name']):
            features = extract_features(row['file_name'])
            X.append(features)
            y.append(row['corrected_emotion'])

    if not X:
        print("No valid data for training.")
        return

    clf = RandomForestClassifier()
    clf.fit(X, y)
    joblib.dump(clf, model_file)
    print(f"✅ Model trained and saved as {model_file}")

if __name__ == "__main__":
    train_emotion_model()

