### 📁 app.py (Main Streamlit App)

import streamlit as st
import joblib
import os
from utils.features import extract_features
from utils.speech_to_text import speech_to_text
from utils.sentiment import get_sentiment
import pandas as pd
import numpy as np

st.title("🎙️ Speech Emotion & Sentiment Analyzer")

# Load model
model = joblib.load("emotion_model.pkl")

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
    if feedback == "No":
        correct_label = st.selectbox("Select correct emotion:", ["happy", "sad", "angry", "neutral"])
        if st.button("Submit Feedback"):
            new_data = np.hstack([features[0], correct_label])
            df = pd.DataFrame([new_data])
            if os.path.exists("feedback.csv"):
                df.to_csv("feedback.csv", mode="a", header=False, index=False)
            else:
                df.to_csv("feedback.csv", index=False)
            st.success("✅ Feedback submitted!")
    import csv
    from datetime import datetime

    def save_result_to_csv(data, filename="results_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

