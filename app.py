### 📁 app.py (Main Streamlit App)

import streamlit as st
import joblib
import os
import csv
from datetime import datetime
import pandas as pd
import numpy as np

from features import extract_features
from speech_to_text import speech_to_text
from sentiment import get_sentiment

st.title("🎙️ Speech Emotion & Sentiment Analyzer")

# Load model
model = joblib.load("emotion_model.pkl")

# CSV Logging function
def save_result_to_csv(data, filename="results_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# File upload
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Extract emotion features
    features = extract_features("temp.wav").reshape(1, -1)
    emotion = model.predict(features)[0]

    # Convert speech to text
    text = speech_to_text("temp.wav")
    sentiment = get_sentiment(text)

    # Display results
    st.subheader("🎧 Transcription")
    st.write(text)

    st.subheader("🎭 Detected Emotion")
    st.write(emotion)

    st.subheader("🧠 Sentiment")
    st.write(f"{sentiment['label']} (Confidence: {sentiment['score']:.2f})")

    # Feedback
    feedback = st.selectbox("Was the emotion prediction correct?", ["Yes", "No"])
    correct_label = None
    if feedback == "No":
        correct_label = st.selectbox("Select correct emotion:", ["happy", "sad", "angry", "neutral"])

    if st.button("Submit Feedback" if feedback == "No" else "Save Result"):
        log_entry = {
            "file_name": uploaded_file.name,
            "transcription": text,
            "predicted_emotion": emotion,
            "sentiment_label": sentiment['label'],
            "sentiment_score": round(sentiment['score'], 2),
            "user_feedback": feedback,
            "corrected_emotion": correct_label if feedback == "No" else "",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_result_to_csv(log_entry)
        st.success("✅ Result logged!")
