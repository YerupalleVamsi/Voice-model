import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from features import extract_features

def train_emotion_and_sentiment_model(log_file="results_log.csv", emotion_model_file="emotion_model.pkl", sentiment_model_file="sentiment_model.pkl"):
    if not os.path.exists(log_file):
        print(f"❌ No training data found at {log_file}")
        return

    df = pd.read_csv(log_file)

    # -------- Train Emotion Model -------- #
    emotion_df = df[(df['user_feedback'] == 'No') & (df['corrected_emotion'].notnull())]
    X_emotion, y_emotion = [], []

    for _, row in emotion_df.iterrows():
        if os.path.exists(row['file_name']):
            features = extract_features(row['file_name'])
            X_emotion.append(features)
            y_emotion.append(row['corrected_emotion'])

    if X_emotion:
        clf_emotion = RandomForestClassifier()
        clf_emotion.fit(X_emotion, y_emotion)
        joblib.dump(clf_emotion, emotion_model_file)
        print(f"✅ Emotion model trained and saved to {emotion_model_file}")
    else:
        print("⚠️ No valid emotion feedback data available for training.")

    # -------- Train Sentiment Model -------- #
    sentiment_df = df[(df['sentiment_feedback'] == 'No') & (df['corrected_sentiment'].notnull()) & (df['transcription'].notnull())]
    X_sentiment = sentiment_df['transcription'].tolist()
    y_sentiment = sentiment_df['corrected_sentiment'].tolist()

    if X_sentiment:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        vectorizer = TfidfVectorizer()
        clf_sentiment = RandomForestClassifier()
        sentiment_pipeline = make_pipeline(vectorizer, clf_sentiment)

        sentiment_pipeline.fit(X_sentiment, y_sentiment)
        joblib.dump(sentiment_pipeline, sentiment_model_file)
        print(f"✅ Sentiment model trained and saved to {sentiment_model_file}")
    else:
        print("⚠️ No valid sentiment feedback data available for training.")

if __name__ == "__main__":
    train_emotion_and_sentiment_model()


