import streamlit as st
import os
import csv
from datetime import datetime
import joblib
import numpy as np
from io import BytesIO
from pydub import AudioSegment
from features import extract_features
from speech_to_text import speech_to_text

# Load models
emotion_model = joblib.load("emotion_model.pkl") if os.path.exists("emotion_model.pkl") else None
sentiment_model = joblib.load("sentiment_model.pkl") if os.path.exists("sentiment_model.pkl") else None

st.title("🎙️ Speech Emotion & Sentiment Analyzer")

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, output_path="temp.wav"):
    audio = AudioSegment.from_file(BytesIO(mp3_file.read()), format="mp3")
    audio.export(output_path, format="wav")

# Save results
def save_result_to_csv(data, filename="results_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# Upload
uploaded_file = st.file_uploader("📤 Upload an audio file", type=["mp3", "wav"])

if uploaded_file:
    temp_path = "temp.wav"
    if uploaded_file.name.endswith(".mp3"):
        convert_mp3_to_wav(uploaded_file, temp_path)
    else:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

    features = extract_features(temp_path).reshape(1, -1)
    transcription = speech_to_text(temp_path)

    # Predictions
    predicted_emotion = emotion_model.predict(features)[0] if emotion_model else "N/A"
    predicted_sentiment = sentiment_model.predict([transcription])[0] if sentiment_model else "neutral"

    st.subheader("🎧 Transcription")
    st.write(transcription)

    st.subheader("🎭 Detected Emotion")
    st.write(predicted_emotion)

    st.subheader("🧠 Sentiment")
    st.write(predicted_sentiment)

    st.markdown("---")
    st.subheader("📝 Feedback")

    emotion_feedback = st.selectbox("Was the emotion prediction correct?", ["Yes", "No"])
    corrected_emotion = None
    if emotion_feedback == "No":
        options = [e for e in ["happy", "sad", "angry", "neutral"] if e != predicted_emotion]
        corrected_emotion = st.selectbox("Select the correct emotion:", options)

    sentiment_feedback = st.selectbox("Was the sentiment prediction correct?", ["Yes", "No"])
    corrected_sentiment = None
    if sentiment_feedback == "No":
        options = [s for s in ["positive", "negative", "neutral"] if s != predicted_sentiment]
        corrected_sentiment = st.selectbox("Select the correct sentiment:", options)

    if st.button("Submit Feedback"):
        log_entry = {
            "file_name": uploaded_file.name,
            "transcription": transcription,
            "predicted_emotion": predicted_emotion,
            "predicted_sentiment": predicted_sentiment,
            "emotion_feedback": emotion_feedback,
            "sentiment_feedback": sentiment_feedback,
            "corrected_emotion": corrected_emotion if emotion_feedback == "No" else "",
            "corrected_sentiment": corrected_sentiment if sentiment_feedback == "No" else "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_result_to_csv(log_entry)
        st.success("✅ Feedback logged!")
