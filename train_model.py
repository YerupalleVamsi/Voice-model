import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import joblib
from features import extract_features

def train_models(log_file="results_log.csv"):
    if not os.path.exists(log_file):
        print("⚠️ No log file found.")
        return

    df = pd.read_csv(log_file)

    # 🎭 Emotion model training
    emotion_df = df[(df["emotion_feedback"] == "No") & (df["corrected_emotion"].notnull())]
    X_emotion, y_emotion = [], []
    for _, row in emotion_df.iterrows():
        if os.path.exists(row["file_name"]):
            try:
                features = extract_features(row["file_name"])
                X_emotion.append(features)
                y_emotion.append(row["corrected_emotion"])
            except:
                continue
    if X_emotion:
        clf = RandomForestClassifier()
        clf.fit(X_emotion, y_emotion)
        joblib.dump(clf, "emotion_model.pkl")
        print("✅ Emotion model trained.")
    else:
        print("⚠️ No valid emotion training data found.")

    # 🧠 Sentiment model training
    sentiment_df = df[(df["sentiment_feedback"] == "No") & (df["corrected_sentiment"].notnull())]
    if not sentiment_df.empty:
        X_sent = sentiment_df["transcription"]
        y_sent = sentiment_df["corrected_sentiment"]
        sentiment_clf = make_pipeline(
            TfidfVectorizer(),
            RandomForestClassifier()
        )
        sentiment_clf.fit(X_sent, y_sent)
        joblib.dump(sentiment_clf, "sentiment_model.pkl")
        print("✅ Sentiment model trained.")
    else:
        print("⚠️ No valid sentiment training data found.")

if __name__ == "__main__":
    train_models()



