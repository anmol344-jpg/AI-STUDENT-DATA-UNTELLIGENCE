# AI Student Decision Intelligence System

One-stop demo: prediction (risk), explainability (SHAP), decision engine (actionable suggestions), simulation (what-if) and Streamlit UI.

Quickstart:
1. python3 -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. python train_model.py        # trains and saves models into /models
4. streamlit run app.py         # open browser to interact

Project structure:
- generate_dataset.py
- data_utils.py
- train_model.py
- prediction.py
- explain.py
- decision_engine.py
- simulation.py
- app.py
- models/ (auto-created)