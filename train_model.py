# train_model.py
# STEP 3: MODEL - train a RandomForest and report metrics
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from generate_dataset import create_dataset
from data_utils import prepare_data

os.makedirs("models", exist_ok=True)

def train_and_save(n=2000):
    df = create_dataset(n)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")
    joblib.dump(model, "models/model.joblib")
    return model, (acc, prec, rec)

if __name__ == "__main__":
    train_and_save(2000)