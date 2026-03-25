# app.py
# Streamlit UI Layer - STEP 7 integrated dashboard
import streamlit as st
import pandas as pd
from prediction import predict_single
from decision_engine import generate_suggestions
from simulation import simulate_change
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Student Decision Intelligence", layout="wide")

st.title("AI Student Decision Intelligence System")
st.markdown("Predict risk, explain why, get actionable suggestions, and run what-if simulations.")

# Input form
with st.sidebar.form("student_form"):
    st.header("Student Inputs")
    attendance = st.slider("Attendance (%)", 30.0, 100.0, 80.0, 0.5)
    study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.5, 0.25)
    previous_marks = st.slider("Previous marks (%)", 0, 100, 70)
    assignments_completed = st.slider("Assignments completed (out of 10)", 0, 10, 7)
    sleep_hours = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.25)
    extracurricular_activity = st.selectbox("Extracurricular activity", (1, 0), index=0, format_func=lambda x: "Yes" if x==1 else "No")
    submitted = st.form_submit_button("Evaluate")

input_features = {
    "attendance": float(attendance),
    "study_hours": float(study_hours),
    "previous_marks": float(previous_marks),
    "assignments_completed": int(assignments_completed),
    "sleep_hours": float(sleep_hours),
    "extracurricular_activity": int(extracurricular_activity)
}

if submitted:
    # Prediction
    result = predict_single(input_features)
    prob = result["probability"]
    risk = result["risk"]
    st.metric(label="Risk (probability of fail/dropout)", value=f"{prob:.2f}", delta="At-risk" if risk==1 else "Safe")

    # Decision suggestions
    st.subheader("AI Coach Suggestions")
    suggestions = generate_suggestions(input_features)
    for s in suggestions:
        st.write("- " + s)

    # What-if simulation sliders
    st.subheader("What-if Simulation (change Attendance / Study hours)")
    with st.expander("Run simulation"):
        att_sim = st.slider("Simulate Attendance (%)", 30.0, 100.0, attendance, 0.5, key="sim_att")
        study_sim = st.slider("Simulate Study hours", 0.0, 12.0, study_hours, 0.25, key="sim_study")
        if st.button("Run Simulation"):
            sim = simulate_change(input_features, attendance=att_sim, study_hours=study_sim)
            st.write("Before:", sim["before"])
            st.write("After:", sim["after"])
            st.write("Suggested actions after change:")
            for s in sim["suggestions"]:
                st.write("- " + s)

    # SHAP explanation (individual)
    st.subheader("Explainability (SHAP)")
    model = joblib.load("models/model.joblib")
    imputer = joblib.load("models/imputer.joblib")
    scaler = joblib.load("models/scaler.joblib")
    feat_df = pd.DataFrame([input_features])
    feat_imputed = pd.DataFrame(imputer.transform(feat_df), columns=feat_df.columns)
    feat_scaled = pd.DataFrame(scaler.transform(feat_imputed), columns=feat_imputed.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feat_scaled)
    # Waterfall / bar chart of contributions
    shap.initjs()
    st.write("Feature contributions to At-Risk score (positive increases risk):")
    fig, ax = plt.subplots()
    vals = shap_values[1][0]
    names = feat_scaled.columns
    contrib = pd.Series(vals, index=names).sort_values(ascending=True)
    contrib.plot.barh(ax=ax, color=["#d62728" if x>0 else "#2ca02c" for x in contrib])
    st.pyplot(fig)

    st.info("One-line explanation: " + ("High risk driven by: " + ", ".join([f"{i}" for i,v in contrib.sort_values(ascending=False).head(3).items()]) ) )