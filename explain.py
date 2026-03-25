# explain.py
# Explainability Engine using SHAP (TreeExplainer for RandomForest)
import shap
import joblib
import pandas as pd

def get_global_explanation(sample_df=None, top_k=6):
    model = joblib.load("models/model.joblib")
    explainer = shap.TreeExplainer(model)
    # For global importance use feature importance via mean(|SHAP|)
    if sample_df is None:
        # create a small sample using train preprocess saved artifacts
        import numpy as np
        sample_df = pd.DataFrame(np.random.normal(size=(100, len(model.feature_importances_))),
                                 columns=[f"f{i}" for i in range(len(model.feature_importances_))])
    shap_vals = explainer.shap_values(sample_df)
    # shap_values[1] corresponds to contribution to class 1 (at-risk)
    mean_abs = pd.Series(np.abs(shap_vals[1]).mean(0), index=sample_df.columns)
    mean_abs = mean_abs.sort_values(ascending=False).head(top_k)
    return mean_abs

def explain_instance(instance_df):
    model = joblib.load("models/model.joblib")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance_df)
    return explainer, shap_values