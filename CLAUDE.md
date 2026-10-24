# Project: Customer Lifetime Value & Churn Predictor

## Context
- B.Tech Computer Engineering ML Lab project (20CP401P), 2-person team.
- Full specification: docs/PROJECT_SPEC.md. Follow it unless this file says otherwise.
- Dataset: Online Retail II (UCI), raw file in data/raw/.
- Must run on a normal laptop (8 GB RAM, no GPU). Keep memory usage low.
- The team is learning. Code must be readable by a student, not clever.

## Working rules
- Work ONLY on the phase I give you. Do not start the next phase on your own.
- At the end of every phase: run the code, show real output, list files created/changed,
  and give a short plain-English summary of what was done and what the results mean.
- Never invent or guess numbers, metrics or results. Only report what the code actually printed.
- If something in the spec is ambiguous or impossible, stop and ask me before choosing.
- If a library is missing, install it with pip and add it to requirements.txt.
- After each phase, make a git commit with message "Phase N: <short description>".

## Code rules
- Python 3.10+. Use pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap,
  matplotlib, seaborn, plotly, streamlit, joblib.
- All settings (paths, cutoff dates, window lengths, random seed, offer cost, response rate)
  live in src/config.py. No hard-coded paths or magic numbers elsewhere.
- random_state = 42 everywhere for reproducibility.
- Reusable logic goes in src/*.py. Notebooks only call functions from src/ and show results.
- Every function gets a short docstring explaining what it does in simple words.
- Add comments explaining WHY, not just what, at important steps (leakage, label creation, SMOTE).
- Save all figures to reports/figures/ as PNG and all result tables to reports/tables/ as CSV.

## Critical ML rules (do not break these)
- NO DATA LEAKAGE: features use only transactions BEFORE the cutoff date;
  labels use only transactions AFTER the cutoff date.
- Fit scalers, feature selectors and SMOTE on TRAINING data only, never on test data.
- Use imblearn Pipeline when combining SMOTE with models so resampling only happens on training folds.
- Do not judge churn models by accuracy alone. Report precision, recall, F1, ROC-AUC, PR-AUC.
- The Streamlit app must use the SAME saved feature builder and models as training.