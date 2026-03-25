# prediction.py
# Prediction Engine: load model + transformers and provide score & class
import joblib
import pandas as pd

MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"
IMPUTER_PATH = "models/imputer.joblib"

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, imputer, scaler

def predict_single(input_dict):
    model, imputer, scaler = load_artifacts()
    df = pd.DataFrame([input_dict])
    X = pd.DataFrame(imputer.transform(df), columns=df.columns)
    Xs = pd.DataFrame(scaler.transform(X), columns=X.columns)
    prob = model.predict_proba(Xs)[0][1]
    pred = int(prob >= 0.5)
    return {"probability": float(prob), "risk": pred, "features": df.iloc[0].to_dict()}