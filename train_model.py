import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from features import extract_features

log_file = "results_log.csv"

# Load Feedback Log
df = pd.read_csv(log_file) if os.path.exists(log_file) else pd.DataFrame()

# ===== Train Emotion Model =====
print("🎭 Training Emotion Model...")
df["final_emotion"] = df.apply(
    lambda row: row["corrected_emotion"] if row["user_feedback_emotion"] == "No" else row["predicted_emotion"], axis=1
)

X_emotion, y_emotion = [], []
for _, row in df.dropna(subset=["file_name", "final_emotion"]).iterrows():
    try:
        if os.path.exists(row["file_name"]):
            features = extract_features(row["file_name"])
            X_emotion.append(features)
            y_emotion.append(row["final_emotion"])
    except Exception as e:
        print(f"❌ Error with {row['file_name']}: {e}")

if X_emotion:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(np.array(X_emotion), y_emotion)
    joblib.dump(model, "emotion_model.pkl")
    print("✅ Emotion model saved.")
else:
    print("⚠️ No emotion data to train.")

# ===== Train Sentiment Model =====
print("🧠 Training Sentiment Model...")
df["final_sentiment"] = df.apply(
    lambda row: row["corrected_sentiment"] if row["user_feedback_sentiment"] == "No" else row["predicted_sentiment"], axis=1
)

sentiment_data = df.dropna(subset=["transcription", "final_sentiment"])

if not sentiment_data.empty:
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(sentiment_data["transcription"])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vec, sentiment_data["final_sentiment"])
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("✅ Sentiment model and vectorizer saved.")
else:
    print("⚠️ No sentiment data to train.")


