"""Churn classifiers: Logistic Regression, CART, Random Forest and XGBoost, with three imbalance options.

Imbalance options:
  "none"          - train on the data as it is
  "class_weight"  - give the minority class more weight in the loss
  "smote"         - SMOTE oversampling INSIDE an imblearn Pipeline, so it only touches training folds

NOTE: in our TRAIN data churners are the MAJORITY (62%). So the minority class is "stays" (0),
and both class weights and SMOTE boost the NON-churners. The effect is expected to be small.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))
import config
from eda import save_matplotlib
from feature_selection import churn_target, make_model_matrix

MODEL_NAMES = ["Logistic Regression", "CART", "Random Forest", "XGBoost"]
IMBALANCE_METHODS = ["none", "class_weight", "smote"]

# Small grids: GridSearchCV only for Random Forest and XGBoost ("light tuning").
PARAM_GRIDS = {
    "Random Forest": {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [5, 20]},
    "XGBoost": {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1],
                "model__n_estimators": [200, 400]},
}


def load_churn_data():
    """Return X_train, y_train, X_test, y_test using the saved encoder and the 8 selected churn features."""
    selected = json.loads(config.SELECTED_FEATURES_JSON.read_text())["churn"]
    encoder = joblib.load(config.SEGMENT_ENCODER)
    train = pd.read_parquet(config.TRAIN_TABLE_PARQUET)
    test = pd.read_parquet(config.TEST_TABLE_PARQUET)
    X_train = make_model_matrix(train, encoder)[selected]
    X_test = make_model_matrix(test, encoder)[selected]
    return X_train, churn_target(train), X_test, churn_target(test), train, test


def make_model(name, imbalance, y_train):
    """Return an untrained classifier; with imbalance='class_weight' the minority class gets more weight."""
    weight = "balanced" if imbalance == "class_weight" else None
    rs = config.RANDOM_STATE
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight=weight)
    if name == "CART":
        # Depth limit keeps the tree readable and stops it memorising the training data.
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight=weight, random_state=rs)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight=weight,
                                      random_state=rs, n_jobs=1)
    if name == "XGBoost":
        # scale_pos_weight = negatives / positives. Churners (1) are the majority, so this is < 1.
        spw = (y_train == 0).sum() / (y_train == 1).sum() if imbalance == "class_weight" else 1.0
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                             random_state=rs, n_jobs=1)
    raise ValueError(f"Unknown model {name}")


def make_pipeline(name, imbalance, y_train):
    """Build an imblearn Pipeline: [scaler] -> [SMOTE] -> model.

    SMOTE sits INSIDE the pipeline, so during cross-validation it only resamples the
    training folds; validation folds and the test set are never resampled.
    """
    steps = []
    if name == "Logistic Regression":
        steps.append(("scaler", StandardScaler()))   # scaling only matters for the linear model
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("model", make_model(name, imbalance, y_train)))
    return Pipeline(steps)


def cv_splitter():
    """Stratified 5-fold split used for every cross-validation on TRAIN."""
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def fit_with_cv(name, imbalance, X_train, y_train):
    """Fit one model variant on TRAIN. Tuned models use GridSearchCV; others get a plain CV score.

    Returns (fitted pipeline, mean CV ROC-AUC on TRAIN, best params or None).
    """
    pipeline = make_pipeline(name, imbalance, y_train)
    if name in PARAM_GRIDS:
        search = GridSearchCV(pipeline, PARAM_GRIDS[name], scoring=config.CV_SCORING, cv=cv_splitter(),
                          n_jobs=config.N_JOBS)
        search.fit(X_train, y_train)
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        return search.best_estimator_, search.best_score_, params
    cv_score = cross_val_score(pipeline, X_train, y_train, scoring=config.CV_SCORING, cv=cv_splitter()).mean()
    return pipeline.fit(X_train, y_train), cv_score, None


def test_metrics(y_true, probability, threshold=config.DECISION_THRESHOLD):
    """Return precision, recall, F1, ROC-AUC, PR-AUC, mean probability and confusion matrix counts."""
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
    return {
        "precision": precision_score(y_true, predicted),
        "recall": recall_score(y_true, predicted),
        "f1": f1_score(y_true, predicted),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "mean_pred_prob": float(np.mean(probability)),
        "actual_churn_rate": float(np.mean(y_true)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def recency_baseline(train_df, test_df):
    """Recency-only reference row: churn if recency > TRAIN median; recency is the ranking score."""
    y = churn_target(test_df)
    predicted = (test_df["recency"] > train_df["recency"].median()).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()
    return {
        "precision": precision_score(y, predicted), "recall": recall_score(y, predicted),
        "f1": f1_score(y, predicted), "roc_auc": roc_auc_score(y, test_df["recency"]),
        "pr_auc": average_precision_score(y, test_df["recency"]),
        "mean_pred_prob": np.nan, "actual_churn_rate": float(y.mean()),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def run_all_models(X_train, y_train, X_test, y_test, train_df, test_df):
    """Train every model x imbalance method; return (comparison table, fitted models, test probabilities)."""
    rows = [{"model": "Recency only (baseline)", "imbalance": "-", "cv_roc_auc_train": np.nan,
             "best_params": "", **recency_baseline(train_df, test_df)}]
    fitted, probabilities = {}, {}
    for name in MODEL_NAMES:
        for imbalance in IMBALANCE_METHODS:
            model, cv_score, params = fit_with_cv(name, imbalance, X_train, y_train)
            probability = model.predict_proba(X_test)[:, 1]
            key = f"{name} | {imbalance}"
            fitted[key], probabilities[key] = model, probability
            rows.append({"model": name, "imbalance": imbalance, "cv_roc_auc_train": cv_score,
                         "best_params": "" if params is None else str(params),
                         **test_metrics(y_test, probability)})
            print(f"done: {key:40s} CV ROC-AUC {cv_score:.3f}  test ROC-AUC {rows[-1]['roc_auc']:.3f}")
    table = pd.DataFrame(rows)
    # The Phase 4 reference (LogReg on the selected features, no imbalance handling) is this row:
    table.loc[(table["model"] == "Logistic Regression") & (table["imbalance"] == "none"), "model"] = \
        "Logistic Regression (Phase 4 reference)"
    return table, fitted, probabilities


def check_leakage_alarm(table):
    """Stop with an error if any model scores suspiciously high on TEST (possible leakage)."""
    suspicious = table[table["roc_auc"] > config.LEAKAGE_ALARM_ROC_AUC]
    if len(suspicious) > 0:
        raise RuntimeError(f"Test ROC-AUC above {config.LEAKAGE_ALARM_ROC_AUC} - check for leakage:\n{suspicious}")
    print(f"Leakage alarm check: no model above test ROC-AUC {config.LEAKAGE_ALARM_ROC_AUC} "
          f"(max {table['roc_auc'].max():.3f})")


def best_variant_per_model(table):
    """For each model, the imbalance method with the highest CV ROC-AUC on TRAIN (test is NOT used)."""
    models = table.dropna(subset=["cv_roc_auc_train"]).copy()
    models["base_model"] = models["model"].str.replace(" (Phase 4 reference)", "", regex=False)
    best = models.loc[models.groupby("base_model")["cv_roc_auc_train"].idxmax()]
    return {row.base_model: f"{row.base_model} | {row.imbalance}" for row in best.itertuples()}


def plot_curves(y_test, probabilities, keys, recency_scores):
    """Save one ROC figure and one precision-recall figure with all given models + the recency baseline."""
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    for key in keys:
        RocCurveDisplay.from_predictions(y_test, probabilities[key], name=key, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y_test, probabilities[key], name=key, ax=ax_pr)
    RocCurveDisplay.from_predictions(y_test, recency_scores, name="Recency only", ax=ax_roc)
    ax_roc.lines[-1].set_linestyle("--")          # dashed line = baseline
    PrecisionRecallDisplay.from_predictions(y_test, recency_scores, name="Recency only", ax=ax_pr)
    ax_pr.lines[-1].set_linestyle("--")
    ax_roc.legend(fontsize=8)
    ax_pr.legend(fontsize=8)
    ax_roc.plot([0, 1], [0, 1], color="grey", linewidth=0.8)
    ax_roc.set_title("ROC curves on TEST")
    ax_pr.set_title("Precision-recall curves on TEST")
    save_matplotlib(fig_roc, "churn_roc_curves")
    save_matplotlib(fig_pr, "churn_pr_curves")
    return fig_roc, fig_pr


def plot_confusion_matrices(y_test, probabilities, keys):
    """Save a grid of confusion matrices (threshold 0.5) for the given models."""
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4))
    for ax, key in zip(np.atleast_1d(axes), keys):
        predicted = (probabilities[key] >= config.DECISION_THRESHOLD).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=["stay", "churn"],
                                                ax=ax, colorbar=False)
        ax.set_title(key, fontsize=9)
    fig.tight_layout()
    save_matplotlib(fig, "churn_confusion_matrices")
    return fig


def lift_table(y_test, probability):
    """Share of all real churners found in the top X% highest-risk customers, and the lift over random."""
    order = np.argsort(-probability)
    y_sorted = np.asarray(y_test)[order]
    rows = []
    for share in config.LIFT_TOP_SHARES:
        n_top = int(round(share * len(y_sorted)))
        captured = y_sorted[:n_top].sum() / y_sorted.sum()
        rows.append({"top_share": f"top {share:.0%}", "customers": n_top,
                     "churn_rate_in_top": y_sorted[:n_top].mean(),
                     "share_of_all_churners_captured": captured, "lift": captured / share})
    return pd.DataFrame(rows)


def plot_gain_curve(y_test, probability, name):
    """Save a cumulative gain chart: % of churners captured vs % of customers contacted."""
    order = np.argsort(-probability)
    y_sorted = np.asarray(y_test)[order]
    contacted = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    captured = np.cumsum(y_sorted) / y_sorted.sum()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(contacted * 100, captured * 100, label=name)
    ax.plot([0, 100], [0, 100], "--", color="grey", label="Random")
    for share in config.LIFT_TOP_SHARES:
        ax.axvline(share * 100, color="lightgrey", linewidth=0.8)
    ax.set_xlabel("% of customers contacted (highest risk first)")
    ax.set_ylabel("% of all churners captured")
    ax.set_title("Cumulative gain chart (TEST)")
    ax.legend()
    save_matplotlib(fig, "churn_gain_chart")
    return fig


def save_churn_model(model, key, features):
    """Save the final churn pipeline with its name and feature list to models/churn_model.pkl."""
    joblib.dump({"model": model, "name": key, "features": features}, config.CHURN_MODEL)
