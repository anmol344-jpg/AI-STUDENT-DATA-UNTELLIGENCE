# generate_dataset.py
# STEP 1: SMART SYNTHETIC DATASET GENERATION
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_dataset(n=2000):
    attendance = np.clip(np.random.normal(loc=80, scale=12, size=n), 30, 100)
    study_hours = np.clip(np.random.normal(loc=3.5, scale=1.8, size=n), 0, 12)
    previous_marks = np.clip(np.random.normal(loc=70, scale=15, size=n), 0, 100)
    assignments_completed = np.clip(np.random.poisson(lam=7, size=n), 0, 10)
    sleep_hours = np.clip(np.random.normal(loc=7.0, scale=1.2, size=n), 3, 10)
    extracurricular_activity = np.random.binomial(1, p=0.38, size=n)

    # Weighted linear propensity (higher => more risk)
    w_att = -0.045
    w_study = -0.25
    w_marks = -0.04
    w_assign = -0.12
    w_sleep = -0.08
    w_extra = -0.18

    linear_score = (
        w_att * attendance +
        w_study * study_hours +
        w_marks * previous_marks +
        w_assign * assignments_completed +
        w_sleep * (7 - sleep_hours) +
        w_extra * (1 - extracurricular_activity)
    )
    linear_score += np.random.normal(scale=0.35, size=n)

    def sigmoid(x): return 1 / (1 + np.exp(-x))
    risk_prob = sigmoid(linear_score * 1.2)
    risk = (risk_prob > np.random.rand(n)).astype(int)

    # Hard rules for realism
    risk[(attendance < 50) | (previous_marks < 40)] = 1

    df = pd.DataFrame({
        "attendance": np.round(attendance, 1),
        "study_hours": np.round(study_hours, 2),
        "previous_marks": np.round(previous_marks, 1),
        "assignments_completed": assignments_completed,
        "sleep_hours": np.round(sleep_hours, 2),
        "extracurricular_activity": extracurricular_activity,
        "risk": risk
    })
    return df

if __name__ == "__main__":
    df = create_dataset(2000)
    print(df.head(8).to_string(index=False))
    print("\nClass distribution:\n", df['risk'].value_counts())