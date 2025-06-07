import speech_recognition as sr

def speech_to_text(audio_file, language="en-US"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Google Speech Recognition service error"
