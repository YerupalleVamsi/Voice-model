import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
from features import extract_features

def train_emotion_model(log_file="results_log.csv", model_file="emotion_model.pkl"):
    if not os.path.exists(log_file):
        print(f"⚠️ No training data found at {log_file}")
        return

    df = pd.read_csv(log_file)
    df = df[(df['user_feedback_emotion'] == 'No') & (df['corrected_emotion'].notnull())]

    X, y = [], []
    for _, row in df.iterrows():
        if os.path.exists(row['file_name']):
            try:
                features = extract_features(row['file_name'])
                X.append(features)
                y.append(row['corrected_emotion'])
            except:
                print(f"⚠️ Skipping file (error extracting features): {row['file_name']}")
    
    if not X:
        print("❌ No valid emotion training data.")
        return

    clf = RandomForestClassifier()
    clf.fit(X, y)
    joblib.dump(clf, model_file)
    print(f"✅ Emotion model trained and saved as {model_file}")


def train_sentiment_model(log_file="results_log.csv", model_file="sentiment_model.pkl", vectorizer_file="tfidf_vectorizer.pkl"):
    if not os.path.exists(log_file):
        print(f"⚠️ No training data found at {log_file}")
        return

    df = pd.read_csv(log_file)
    df = df[(df['user_feedback_sentiment'] == 'No') & (df['corrected_sentiment'].notnull())]

    if df.empty:
        print("❌ No valid sentiment training data.")
        return

    X_texts = df['transcription']
    y = df['corrected_sentiment']

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    print(f"✅ Sentiment model trained and saved as {model_file}")
    print(f"✅ Vectorizer saved as {vectorizer_file}")


if __name__ == "__main__":
    train_emotion_model()
    train_sentiment_model()


