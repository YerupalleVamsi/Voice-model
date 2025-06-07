
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from features import extract_features
from speech_to_text import transcribe_audio
import os

log_file = "results_log.csv"

if not os.path.exists(log_file):
    print("No feedback to retrain on.")
    exit()

data = pd.read_csv(log_file).dropna(subset=["correct_emotion", "correct_sentiment"])

print(f"Loaded {len(data)} feedback entries.")

# Emotion model retraining
X_emotion = np.array([extract_features(row["audio_file"]) for _, row in data.iterrows()])
y_emotion = data["correct_emotion"]

emotion_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
emotion_model.fit(X_emotion, y_emotion)
joblib.dump(emotion_model, "emotion_model.pkl")
print("Updated emotion_model.pkl")

# Sentiment model retraining
texts = data["transcribed_text"].fillna("")
y_sentiment = data["correct_sentiment"]

vectorizer = TfidfVectorizer()
X_sentiment = vectorizer.fit_transform(texts)

sentiment_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
sentiment_model.fit(X_sentiment, y_sentiment)

joblib.dump(sentiment_model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Updated sentiment_model.pkl and tfidf_vectorizer.pkl")

# Optional: Evaluation print
y_pred_emotion = emotion_model.predict(X_emotion)
y_pred_sentiment = sentiment_model.predict(X_sentiment)

print("Emotion classification report:")
print(classification_report(y_emotion, y_pred_emotion))

print("Sentiment classification report:")
print(classification_report(y_sentiment, y_pred_sentiment))
