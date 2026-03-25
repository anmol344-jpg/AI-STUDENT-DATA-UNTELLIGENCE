# simulation.py
# Simulation Engine: simple deterministic simulation using the prediction pipeline
from prediction import predict_single
from decision_engine import generate_suggestions
import copy

def simulate_change(input_features, attendance=None, study_hours=None):
    before = predict_single(input_features)
    new_features = copy.deepcopy(input_features)
    if attendance is not None:
        new_features["attendance"] = attendance
    if study_hours is not None:
        new_features["study_hours"] = study_hours
    after = predict_single(new_features)
    suggestions = generate_suggestions(new_features)
    return {"before": before, "after": after, "suggestions": suggestions}