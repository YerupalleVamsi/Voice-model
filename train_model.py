import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load feedback
data = pd.read_csv("results_log.csv")

# Clean rows with missing values
data = data.dropna()

# Train sentiment model
X_text = data['Transcription']
y_sentiment = data['Corrected Sentiment']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_text)

sentiment_model = LogisticRegression()
sentiment_model.fit(X_vec, y_sentiment)

# Save sentiment model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(sentiment_model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Train emotion model
y_emotion = data['Corrected Emotion']
emotion_model = LogisticRegression()
emotion_model.fit(X_vec, y_emotion)

with open("emotion_model.pkl", "wb") as f:
    pickle.dump(emotion_model, f)

print("✅ Models retrained and saved.")



