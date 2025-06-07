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
import subprocess
import time

# ========== Auto Retrain Logic ==========
log_file = "results_log.csv"
train_script = "train_model.py"
last_run_file = "last_model_update.txt"

def should_retrain():
    if not os.path.exists(log_file):
        return False
    if not os.path.exists(last_run_file):
        return True
    with open(last_run_file, "r") as f:
        last_time = float(f.read().strip())
    return os.path.getmtime(log_file) > last_time

def retrain_models():
    subprocess.run(["python", train_script])
    with open(last_run_file, "w") as f:
        f.write(str(time.time()))

# Run training if feedback has been updated
if should_retrain():
    st.info("🔄 Updating models with latest feedback...")
    retrain_models()

# ========== Load Models ==========
try:
    emotion_model = joblib.load("emotion_model.pkl")
except FileNotFoundError:
    st.warning("⚠️ No trained emotion model found.")
    emotion_model = None

try:
    sentiment_model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.warning("⚠️ No trained sentiment model found.")
    sentiment_model = None
    vectorizer = None

# ========== App Title ==========
st.title("🎙️ Speech Emotion & Sentiment Analyzer")

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, output_path="temp.wav"):
    audio = AudioSegment.from_file(BytesIO(mp3_file.read()), format="mp3")
    audio.export(output_path, format="wav")

# Save Feedback
def save_result_to_csv(data, filename="results_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# Upload Audio
uploaded_file = st.file_uploader("📤 Upload an audio file", type=["mp3", "wav"])

if uploaded_file:
    temp_path = "temp.wav"

    if uploaded_file.name.endswith(".mp3"):
        convert_mp3_to_wav(uploaded_file, temp_path)
    else:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

    # Extract features and predict emotion
    features = extract_features(temp_path).reshape(1, -1)
    predicted_emotion = emotion_model.predict(features)[0] if emotion_model else "N/A"

    # Transcribe audio
    transcription = speech_to_text(temp_path)

    # Predict sentiment
    if sentiment_model and vectorizer:
        text_vector = vectorizer.transform([transcription])
        predicted_sentiment = sentiment_model.predict(text_vector)[0]
        sentiment_score = max(sentiment_model.predict_proba(text_vector)[0])
    else:
        predicted_sentiment = "unknown"
        sentiment_score = 0.0

    # Show results
    st.subheader("🎧 Transcription")
    st.write(transcription)

    st.subheader("🎭 Detected Emotion")
    st.write(predicted_emotion)

    st.subheader("🧠 Sentiment")
    st.write(f"{predicted_sentiment} (Confidence: {sentiment_score:.2f})")

    # Feedback
    st.subheader("📝 Feedback")
    
    # Emotion
    feedback_emotion = st.selectbox("Was the emotion prediction correct?", ["Yes", "No"])
    corrected_emotion = ""
    if feedback_emotion == "No":
        emotion_options = ["happy", "sad", "angry", "neutral"]
        emotion_options = [e for e in emotion_options if e != predicted_emotion]
        corrected_emotion = st.selectbox("Select the correct emotion:", emotion_options)

    # Sentiment
    feedback_sentiment = st.selectbox("Was the sentiment prediction correct?", ["Yes", "No"])
    corrected_sentiment = ""
    if feedback_sentiment == "No":
        sentiment_options = ["positive", "negative", "neutral"]
        sentiment_options = [s for s in sentiment_options if s != predicted_sentiment]
        corrected_sentiment = st.selectbox("Select the correct sentiment:", sentiment_options)

    # Submit
    if st.button("Submit Feedback"):
        log_entry = {
            "file_name": uploaded_file.name,
            "transcription": transcription,
            "predicted_emotion": predicted_emotion,
            "predicted_sentiment": predicted_sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "user_feedback_emotion": feedback_emotion,
            "corrected_emotion": corrected_emotion if feedback_emotion == "No" else predicted_emotion,
            "user_feedback_sentiment": feedback_sentiment,
            "corrected_sentiment": corrected_sentiment if feedback_sentiment == "No" else predicted_sentiment,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_result_to_csv(log_entry)
        st.success("✅ Feedback logged! Models will update automatically.")


