"""Two-part ("hurdle") spend model.

About half of the customers spend 0 in the next 6 months. A single regression on log1p(spend)
has to squeeze those zeros and the real spenders into one line, which pulls the spenders' predictions
far down. The two-part model splits the question:

  Part 1 (classifier):  p = P(future_spend > 0)
  Part 2 (regressor):   amount = expected spend IF the customer spends, trained on spenders only,
                        on log1p(spend), back-transformed with Duan's smearing (learned on spenders only)
  Expected spend = p * amount, capped at the TRAIN maximum.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent))
import config
from spend_models import historical_rate_baseline, smearing_factor, spend_metrics

PART_NAMES = ["Logistic/Linear", "XGBoost"]

# Small grids for the XGBoost parts ("light tuning").
XGB_GRID = {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1], "model__n_estimators": [200, 400]}


def make_classifier(name):
    """Part 1 pipeline: predicts whether a customer will spend anything (1) or not (0)."""
    if name == "Logistic/Linear":
        return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
    return Pipeline([("model", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, eval_metric="logloss",
                                             random_state=config.RANDOM_STATE, n_jobs=1))])


def make_regressor(name):
    """Part 2 pipeline: predicts log1p(spend) for customers who do spend."""
    if name == "Logistic/Linear":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    return Pipeline([("model", XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                            colsample_bytree=0.8, random_state=config.RANDOM_STATE, n_jobs=1))])


def kfold():
    """Shuffled 5-fold split for regression and for the combined model."""
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def stratified_kfold():
    """Stratified 5-fold split for the spend > 0 classifier."""
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def amount_gbp(regressor, X, smearing):
    """Turn the part-2 log prediction into GBP with Duan's smearing: exp(pred) * S - 1, at least 0."""
    return np.clip(np.exp(regressor.predict(X)) * smearing - 1, 0, None)


def evaluate_part1(X_train, spent):
    """Tune (XGBoost only) and CV-score both classifiers on TRAIN. Returns (table, tuned pipelines)."""
    rows, pipelines = [], {}
    for name in PART_NAMES:
        pipeline = make_classifier(name)
        if name == "XGBoost":
            search = GridSearchCV(pipeline, XGB_GRID, scoring="roc_auc", cv=stratified_kfold(), n_jobs=config.N_JOBS)
            pipeline = pipeline.set_params(**search.fit(X_train, spent).best_params_)
        score = cross_val_score(pipeline, X_train, spent, scoring="roc_auc", cv=stratified_kfold()).mean()
        pipelines[name] = pipeline
        rows.append({"part": "1: P(spend > 0)", "model": name, "cv_roc_auc": score,
                     "best_params": tuned_params(pipeline) if name == "XGBoost" else ""})
    return pd.DataFrame(rows), pipelines


def tuned_params(pipeline):
    """Return the grid-searched XGBoost settings of a pipeline as a short text."""
    params = pipeline.get_params()
    return str({key.replace("model__", ""): params[key] for key in XGB_GRID})


def choose_parts(part1_table, part2_table):
    """Part 1 by highest CV ROC-AUC, part 2 by lowest CV RMSE in GBP on spenders (TRAIN only)."""
    part1 = part1_table.loc[part1_table["cv_roc_auc"].idxmax(), "model"]
    part2 = part2_table.loc[part2_table["cv_rmse_gbp"].idxmin(), "model"]
    return part1, part2


def evaluate_part2(X_spenders, y_spenders_log):
    """Tune (XGBoost only) and CV-score both regressors on TRAIN SPENDERS, in GBP with smearing per fold."""
    y_log = np.asarray(y_spenders_log)
    y_gbp = np.expm1(y_log)
    rows, pipelines = [], {}
    for name in PART_NAMES:
        pipeline = make_regressor(name)
        if name == "XGBoost":
            search = GridSearchCV(pipeline, XGB_GRID, scoring="neg_root_mean_squared_error", cv=kfold(),
                                  n_jobs=config.N_JOBS)
            pipeline = pipeline.set_params(**search.fit(X_spenders, y_log).best_params_)
        oof = np.zeros(len(y_log))
        factors = []
        for tr_idx, va_idx in kfold().split(X_spenders):
            model = clone(pipeline).fit(X_spenders.iloc[tr_idx], y_log[tr_idx])
            factor = smearing_factor(model, X_spenders.iloc[tr_idx], y_log[tr_idx])   # fold-train spenders only
            factors.append(factor)
            oof[va_idx] = amount_gbp(model, X_spenders.iloc[va_idx], factor)
        pipelines[name] = pipeline
        rows.append({"part": "2: amount | spend > 0", "model": name,
                     "cv_rmse_gbp": float(np.sqrt(mean_squared_error(y_gbp, oof))),
                     "cv_mae_gbp": mean_absolute_error(y_gbp, oof), "cv_r2_gbp": r2_score(y_gbp, oof),
                     "cv_mean_predicted": float(oof.mean()), "cv_mean_actual": float(y_gbp.mean()),
                     "mean_smearing_factor": float(np.mean(factors)),
                     "best_params": tuned_params(pipeline) if name == "XGBoost" else ""})
    return pd.DataFrame(rows), pipelines


def fit_hurdle(classifier, regressor, X, y_gbp):
    """Fit part 1 on all given customers and part 2 on the spenders among them. Returns a model bundle."""
    y_gbp = np.asarray(y_gbp)
    spent = (y_gbp > 0).astype(int)
    part1 = clone(classifier).fit(X, spent)
    spenders = y_gbp > 0
    y_log = np.log1p(y_gbp[spenders])
    part2 = clone(regressor).fit(X[spenders], y_log)
    # Smearing learned on (training) spenders only: their log errors are much smaller and more even.
    factor = smearing_factor(part2, X[spenders], y_log)
    return {"type": "hurdle", "classifier": part1, "regressor": part2, "smearing_factor": factor}


def predict_hurdle(bundle, X, cap=None):
    """Expected spend in GBP = P(spend > 0) * amount if spending, capped at `cap`."""
    probability = bundle["classifier"].predict_proba(X)[:, 1]
    amount = amount_gbp(bundle["regressor"], X, bundle["smearing_factor"])
    return np.clip(probability * amount, 0, cap)


def cross_validate_hurdle(classifier, regressor, X_train, y_train_gbp):
    """Out-of-fold CV of the whole two-part model on TRAIN, in GBP (same folds as the Phase 6 models)."""
    y = np.asarray(y_train_gbp)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in kfold().split(X_train):
        bundle = fit_hurdle(classifier, regressor, X_train.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = predict_hurdle(bundle, X_train.iloc[va_idx])
    return {"cv_rmse_gbp": float(np.sqrt(mean_squared_error(y, oof))), "cv_mae_gbp": mean_absolute_error(y, oof),
            "cv_r2_gbp": r2_score(y, oof), "cv_mean_predicted": float(oof.mean())}


def test_comparison(hurdle_bundle, plain_model, X_test, y_test_gbp, test_df, cap):
    """TEST metrics (GBP) for the two-part model, plain Linear Regression and the historical-rate baseline."""
    predictions = {
        "Two-part model": predict_hurdle(hurdle_bundle, X_test, cap),
        "Plain Linear Regression (Phase 6 final)": np.minimum(np.clip(np.expm1(plain_model.predict(X_test)), 0, None), cap),
        "Historical-rate baseline": historical_rate_baseline(test_df),
    }
    rows = [{"model": name, "n_capped": int((pred >= cap).sum()) if "baseline" not in name else 0,
             **spend_metrics(y_test_gbp, pred)} for name, pred in predictions.items()]
    return pd.DataFrame(rows), predictions


def save_hurdle_model(bundle, features, cap, part_names):
    """Save the two-part model as the spend model (same file the app loads)."""
    joblib.dump({**bundle, "name": f"Two-part model ({part_names})", "features": features, "cap_gbp": cap},
                config.SPEND_MODEL)
