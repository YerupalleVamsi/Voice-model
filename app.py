import streamlit as st
import os
import csv
import joblib
import numpy as np
from datetime import datetime
from features import extract_features
from speech_to_text import speech_to_text
from sentiment import get_sentiment

st.title("🎙️ Speech Emotion & Sentiment Analyzer")

model_path = "emotion_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.warning("❌ No emotion model found. Please run `train_model.py` first.")
    st.stop()

def save_result_to_csv(data, filename="results_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file:
    temp_audio_path = f"uploads/{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
    os.makedirs("uploads", exist_ok=True)
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())

    features = extract_features(temp_audio_path).reshape(1, -1)
    emotion = model.predict(features)[0]

    text = speech_to_text(temp_audio_path)
    sentiment = get_sentiment(text)

    st.subheader("🎧 Transcription")
    st.write(text)

    st.subheader("🎭 Detected Emotion")
    st.write(emotion)

    st.subheader("🧠 Sentiment")
    st.write(f"{sentiment['label']} (Confidence: {sentiment['score']:.2f})")

    feedback = st.selectbox("Was the emotion prediction correct?", ["Yes", "No"])
    correct_label = None
    if feedback == "No":
        correct_label = st.selectbox("Select correct emotion:", ["happy", "sad", "angry", "neutral"])

    if st.button("Submit Feedback" if feedback == "No" else "Save Result"):
        log_entry = {
            "file_name": temp_audio_path,
            "transcription": text,
            "predicted_emotion": emotion,
            "sentiment_label": sentiment['label'],
            "sentiment_score": round(sentiment['score'], 2),
            "user_feedback": feedback,
            "corrected_emotion": correct_label if feedback == "No" else "",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_result_to_csv(log_entry)
        st.success("✅ Result logged! You can now retrain the model with `train_model.py`.")
