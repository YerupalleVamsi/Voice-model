
import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib
from features import extract_features
from speech_to_text import transcribe_audio
from pydub import AudioSegment
import uuid
import librosa
import tempfile

st.set_page_config(page_title="Speech Emotion & Sentiment Analyzer", layout="centered")
st.markdown("<style>body { font-family: Arial, sans-serif; }</style>", unsafe_allow_html=True)

emotion_model = joblib.load("emotion_model.pkl")
sentiment_model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def save_feedback(audio, text, predicted_emotion, predicted_sentiment, correct_emotion, correct_sentiment):
    feedback = pd.DataFrame([{
        "audio_file": audio,
        "transcribed_text": text,
        "predicted_emotion": predicted_emotion,
        "predicted_sentiment": predicted_sentiment,
        "correct_emotion": correct_emotion,
        "correct_sentiment": correct_sentiment
    }])
    feedback.to_csv("results_log.csv", mode='a', header=not os.path.exists("results_log.csv"), index=False)

st.title("🎤 Speech Emotion & Sentiment Analyzer")
uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        if uploaded_file.type == "audio/mpeg":
            sound = AudioSegment.from_mp3(uploaded_file)
        else:
            sound = AudioSegment.from_file(uploaded_file)
        sound.export(temp_wav_file.name, format="wav")
        st.audio(temp_wav_file.name)

        features = extract_features(temp_wav_file.name)
        try:
            emotion = emotion_model.predict([features])[0]
            emotion_probs = emotion_model.predict_proba([features])[0]
        except Exception as e:
            st.error(f"Emotion model error: {e}")
            emotion, emotion_probs = "Unknown", []

        text = transcribe_audio(temp_wav_file.name)
        try:
            tfidf_input = vectorizer.transform([text])
            sentiment = sentiment_model.predict(tfidf_input)[0]
            sentiment_probs = sentiment_model.predict_proba(tfidf_input)[0]
        except Exception as e:
            st.error(f"Sentiment model error: {e}")
            sentiment, sentiment_probs = "Unknown", []

        st.subheader("🔊 Predictions")
        st.write(f"**Emotion:** {emotion} — {dict(zip(emotion_model.classes_, emotion_probs))}")
        st.write(f"**Sentiment:** {sentiment} — {dict(zip(sentiment_model.classes_, sentiment_probs))}")
        st.write(f"**Transcribed Text:** {text}")

        st.subheader("✍️ Feedback")
        corrected_emotion = st.selectbox("Correct Emotion", [""] + [e for e in emotion_model.classes_ if e != emotion])
        corrected_sentiment = st.selectbox("Correct Sentiment", [""] + [s for s in sentiment_model.classes_ if s != sentiment])
        if st.button("Submit Feedback"):
            save_feedback(temp_wav_file.name, text, emotion, sentiment, corrected_emotion, corrected_sentiment)
            st.success("Feedback submitted. Thank you!")
