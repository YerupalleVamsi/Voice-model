# sentiment.py
from transformers import pipeline

# Use a pre-trained sentiment-analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def get_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return {"label": result['label'].lower(), "score": result['score']}
