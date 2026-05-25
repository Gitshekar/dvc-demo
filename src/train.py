import pandas as pd
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train():
    print("[train] Loading features...")
    df = pd.read_csv("data/processed/features.csv")

    # Split features and target
    X = df.drop('pass', axis=1)
    y = df['pass']

    # Train/test split — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save model and split info
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Save test data for evaluate step
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"[train] Trained on {len(X_train)} samples")
    print("[train] Saved → models/model.pkl ✓")

if __name__ == "__main__":
    train()