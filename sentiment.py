from transformers import pipeline
import os

def get_sentiment(text):
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = sentiment_analyzer(text)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }
    except Exception as e:
        return {
            "label": "unknown",
            "score": 0.0
        }
