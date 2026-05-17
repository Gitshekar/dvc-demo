import pandas as pd
import os

def prepare_data():
    print("[prepare] Loading raw data...")
    df = pd.read_csv("data/raw/students.csv")

    print(f"[prepare] Raw shape: {df.shape}")

    # 1. Drop duplicates
    df = df.drop_duplicates()

    # 2. Drop rows with missing values
    df = df.dropna()

    # 3. Remove student_id (not useful for ML)
    df = df.drop('student_id', axis=1)

    # 4. Clip outliers — attendance must be 0-100
    df['attendance_pct'] = df['attendance_pct'].clip(0, 100)
    df['sleep_hours']    = df['sleep_hours'].clip(0, 24)

    # 5. Save clean data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean.csv", index=False)

    print(f"[prepare] Clean shape: {df.shape}")
    print("[prepare] Saved → data/processed/clean.csv ✓")

if __name__ == "__main__":
    prepare_data()