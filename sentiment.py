from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def get_sentiment(text):
    if not text:
        return {"label": "NEUTRAL", "score": 0.0}
    return sentiment_analyzer(text)[0]
