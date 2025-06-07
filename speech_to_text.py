import whisper

# Load model once
model = whisper.load_model("base")  # use "small", "medium", or "large" for better accuracy

def speech_to_text(audio_file):
    try:
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"Whisper error: {str(e)}"

