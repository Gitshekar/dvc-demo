import pandas as pd
import numpy as np

np.random.seed(42)
n = 50

df = pd.DataFrame({
    'student_id'    : range(1, n+1),
    'hours_studied' : np.random.randint(1, 10, n),
    'attendance_pct': np.random.randint(40, 100, n),
    'prev_score'    : np.random.randint(30, 95, n),
    'sleep_hours'   : np.random.randint(4, 10, n),
    'extra_curricular': np.random.randint(0, 2, n),  # 0 or 1
})

# Create target: pass if score_formula > 60
score = (
    df['hours_studied'] * 4 +
    df['attendance_pct'] * 0.3 +
    df['prev_score'] * 0.5 +
    np.random.normal(0, 5, n)
)
df['pass'] = (score > 65).astype(int)

df.to_csv('data/raw/students.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df.head())