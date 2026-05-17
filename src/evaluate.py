import pandas as pd
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report
)

def evaluate():
    print("[evaluate] Loading model and test data...")

    model  = joblib.load("models/model.pkl")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "f1_score"  : round(f1_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred), 4),
        "recall"    : round(recall_score(y_test, y_pred), 4),
    }

    # Save metrics as JSON (DVC will track this!)
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[evaluate] ── Results ──────────────────")
    for k, v in metrics.items():
        print(f"[evaluate]   {k:12s}: {v:.4f}")
    print("[evaluate] Saved → metrics/scores.json ✓")

if __name__ == "__main__":
    evaluate()
