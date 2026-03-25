# data_utils.py
# STEP 2: DATA PREPROCESSING - minimal & reproducible
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["risk"])
    y = df["risk"].astype(int)
    # Simple imputation (if any)
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Save transformer objects
    joblib.dump(imputer, "models/imputer.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    return X_train, X_test, y_train, y_test