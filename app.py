import streamlit as st
import os
import csv
from datetime import datetime
import joblib
import numpy as np
from pydub import AudioSegment
from features import extract_features
from speech_to_text import speech_to_text
from sentiment import get_sentiment

# Set title
st.title("🎙️ Speech Emotion & Sentiment Analyzer")

# Try to load model
try:
    model = joblib.load("emotion_model.pkl")
except FileNotFoundError:
    st.warning("⚠️ No trained model found. Predictions will be unavailable until trained.")
    model = None

# Convert MP3 to WAV
from io import BytesIO

def convert_mp3_to_wav(mp3_file, output_path="temp.wav"):
    audio = AudioSegment.from_file(BytesIO(mp3_file.read()), format="mp3")
    audio.export(output_path, format="wav")


# Save results to CSV
def save_result_to_csv(data, filename="results_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# Upload audio
uploaded_file = st.file_uploader("📤 Upload an audio file", type=["mp3", "wav"])

if uploaded_file:
    # Save uploaded file as .wav
    temp_path = "temp.wav"
    if uploaded_file.name.endswith(".mp3"):
        convert_mp3_to_wav(uploaded_file, temp_path)
    else:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

    # Extract features and make prediction
    features = extract_features(temp_path).reshape(1, -1)
    emotion = model.predict(features)[0] if model else "N/A"

    # Speech to text + sentiment
    text = speech_to_text(temp_path)
    sentiment = get_sentiment(text)

    # Display
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
        correct_label = st.selectbox("Select the correct emotion:", ["happy", "sad", "angry", "neutral"])

    if st.button("Submit Feedback" if feedback == "No" else "Save Result"):
        log_entry = {
            "file_name": uploaded_file.name,
            "transcription": text,
            "predicted_emotion": emotion,
            "sentiment_label": sentiment["label"],
            "sentiment_score": round(sentiment["score"], 2),
            "user_feedback": feedback,
            "corrected_emotion": correct_label if feedback == "No" else "",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_result_to_csv(log_entry)
        st.success("✅ Logged successfully!")

