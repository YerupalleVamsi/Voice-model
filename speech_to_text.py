import whisper

model = whisper.load_model("base")  # or "small", "medium", "large"

def speech_to_text(audio_file, language="en"):
    try:
        result = model.transcribe(audio_file, language=language)
        return result["text"]
    except Exception as e:
        return f"Whisper error: {str(e)}"
