# train_model.py
import pandas as pd
import numpy as np
import os
from features import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the feedback log
df = pd.read_csv("results_log.csv")

# Use corrected emotion if available, else fallback to predicted
df["label"] = df.apply(
    lambda row: row["corrected_emotion"] if pd.notna(row["corrected_emotion"]) and row["corrected_emotion"] != "" else row["predicted_emotion"],
    axis=1
)

X = []
y = []

for i, row in df.iterrows():
    try:
        features = extract_features(f"temp.wav")  # You should save and use original uploaded files, not "temp.wav"
        X.append(features)
        y.append(row["label"])
    except Exception as e:
        print(f"Skipping file due to error: {e}")

X = np.array(X)
y = np.array(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "emotion_model.pkl")
print("Model saved as emotion_model.pkl")
