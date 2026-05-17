import pandas as pd
import os

def featurize():
    print("[featurize] Loading clean data...")
    df = pd.read_csv("data/processed/clean.csv")

    # Feature 1: study efficiency score
    df['study_score'] = df['hours_studied'] * df['attendance_pct'] / 100

    # Feature 2: is student high performer in past?
    df['high_prev_score'] = (df['prev_score'] > 70).astype(int)

    # Feature 3: well rested (7-9 hours is ideal)
    df['well_rested'] = (
        (df['sleep_hours'] >= 7) & (df['sleep_hours'] <= 9)
    ).astype(int)

    # Feature 4: overall engagement index
    df['engagement'] = (
        df['hours_studied'] * 0.4 +
        df['attendance_pct'] * 0.4 +
        df['extra_curricular'] * 10 +
        df['well_rested'] * 5
    )

    # Save featured data
    df.to_csv("data/processed/features.csv", index=False)

    print(f"[featurize] Features added: {list(df.columns)}")
    print("[featurize] Saved → data/processed/features.csv ✓")

if __name__ == "__main__":
    featurize()