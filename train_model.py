# train_model.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load feedback log
df = pd.read_csv("results_log.csv")

# Drop incomplete rows
df.dropna(subset=["Transcription", "Corrected Sentiment", "Corrected Emotion"], inplace=True)

# TF-IDF Vectorization
X_text = df["Transcription"]
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_text)

# Train Sentiment Model
y_sentiment = df["Corrected Sentiment"]
sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_vec, y_sentiment)

# Train Emotion Model
y_emotion = df["Corrected Emotion"]
emotion_model = LogisticRegression(max_iter=1000)
emotion_model.fit(X_vec, y_emotion)

# Save models and vectorizer
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(sentiment_model, f)

with open("emotion_model.pkl", "wb") as f:
    pickle.dump(emotion_model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Models retrained and saved.")
