import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from features import extract_features
import os

log_file = "results_log.csv"

# Load feedback data
if not os.path.exists(log_file):
    print("⚠️ No feedback file found.")
    exit()

df = pd.read_csv(log_file)

# ========== EMOTION MODEL TRAINING ==========
print("🎭 Training Emotion Model...")

# Use corrected emotion if user said prediction was wrong
df["final_emotion"] = df.apply(
    lambda row: row["corrected_emotion"] if row["user_feedback_emotion"] == "No" else row["predicted_emotion"], axis=1
)

emotion_records = df[["file_name", "final_emotion"]].dropna()

X_emotion, y_emotion = [], []

for _, row in emotion_records.iterrows():
    audio_path = row["file_name"]
    if not os.path.exists(audio_path):
        print(f"⚠️ File missing: {audio_path}")
        continue
    try:
        features = extract_features(audio_path)
        X_emotion.append(features)
        y_emotion.append(row["final_emotion"])
    except Exception as e:
        print(f"⚠️ Feature extraction failed for {audio_path}: {e}")

if X_emotion:
    emotion_model = RandomForestClassifier(n_estimators=100, random_state=42)
    emotion_model.fit(np.array(X_emotion), y_emotion)
    joblib.dump(emotion_model, "emotion_model.pkl")
    print("✅ Emotion model saved.")
else:
    print("❌ No valid emotion data to train on.")

# ========== SENTIMENT MODEL TRAINING ==========
print("🧠 Training Sentiment Model...")

# Use corrected sentiment if user said prediction was wrong
df["final_sentiment"] = df.apply(
    lambda row: row["corrected_sentiment"] if row["user_feedback_sentiment"] == "No" else row["predicted_sentiment"], axis=1
)

sentiment_df = df[["transcription", "final_sentiment"]].dropna()

if not sentiment_df.empty:
    X_text = sentiment_df["transcription"]
    y_sentiment = sentiment_df["final_sentiment"]

    tfidf = TfidfVectorizer()
    X_vec = tfidf.fit_transform(X_text)

    sentiment_model = RandomForestClassifier(n_estimators=100, random_state=42)
    sentiment_model.fit(X_vec, y_sentiment)

    joblib.dump(sentiment_model, "sentiment_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")

    print("✅ Sentiment model and vectorizer saved.")
else:
    print("❌ No valid sentiment data to train on.")

