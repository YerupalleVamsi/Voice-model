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
from sentiment import get_sentiment

st.title("🎙️ Speech Emotion & Sentiment Analyzer")

# Load model
try:
    model = joblib.load("emotion_model.pkl")
except FileNotFoundError:
    st.warning("⚠️ No trained model found. Predictions will be unavailable until trained.")
    model = None

# MP3 to WAV conversion
def convert_mp3_to_wav(mp3_file, output_path="temp.wav"):
    audio = AudioSegment.from_file(BytesIO(mp3_file.read()), format="mp3")
    audio.export(output_path, format="wav")

# Save logs
def save_result_to_csv(data, filename="results_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# Upload file
uploaded_file = st.file_uploader("📤 Upload an audio file", type=["mp3", "wav"])

if uploaded_file:
    temp_path = "temp.wav"
    if uploaded_file.name.endswith(".mp3"):
        convert_mp3_to_wav(uploaded_file, temp_path)
    else:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

    # Predict
    features = extract_features(temp_path).reshape(1, -1)
    emotion = model.predict(features)[0] if model else "N/A"
    text = speech_to_text(temp_path)
    sentiment = get_sentiment(text)

    # Display predictions
    st.subheader("🎧 Transcription")
    st.write(text)

    st.subheader("🎭 Predicted Emotion")
    st.write(emotion)

    st.subheader("🧠 Sentiment")
    st.write(f"{sentiment['label']} (Confidence: {sentiment['score']:.2f})")

    # Emotion Feedback
    st.markdown("### 📝 Feedback")
    feedback_emotion = st.selectbox("Was the emotion prediction correct?", ["Yes", "No"])
    corrected_emotion = None
    if feedback_emotion == "No":
        emotion_options = [e for e in ["happy", "sad", "angry", "neutral"] if e != emotion]
        corrected_emotion = st.selectbox("Select the correct emotion:", emotion_options)

    # Sentiment Feedback
    feedback_sentiment = st.selectbox("Was the sentiment prediction correct?", ["Yes", "No"])
    corrected_sentiment = None
    if feedback_sentiment == "No":
        sentiment_options = [s for s in ["positive", "negative", "neutral"] if s != sentiment["label"]]
        corrected_sentiment = st.selectbox("Select the correct sentiment:", sentiment_options)

    # Save feedback
    if st.button("Submit Feedback"):
        log_entry = {
            "file_name": uploaded_file.name,
            "transcription": text,
            "predicted_emotion": emotion,
            "sentiment_label": sentiment["label"],
            "sentiment_score": round(sentiment["score"], 2),
            "user_feedback": feedback_emotion,
            "corrected_emotion": corrected_emotion if feedback_emotion == "No" else "",
            "sentiment_feedback": feedback_sentiment,
            "corrected_sentiment": corrected_sentiment if feedback_sentiment == "No" else "",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_result_to_csv(log_entry)
        st.success("✅ Feedback saved successfully!")


    
