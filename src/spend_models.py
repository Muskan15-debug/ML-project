"""Future spend regression: Linear Regression, Random Forest and XGBoost on log1p(future_spend).

Why log1p: about half of the customers spend 0 and a few spend over 100,000 GBP. On the raw
scale the error is dominated by those few giants. log1p squeezes the long tail and keeps 0 as 0.
Predictions are turned back into GBP, and all reported errors are in GBP.

Two GBP corrections are available:
- Duan's smearing: expm1 of a log prediction estimates the typical (median-like) customer,
  not the average, so GBP predictions come out too low on average. Smearing multiplies
  exp(prediction) by S = mean(exp(residuals)), learned on TRAIN only.
- Cap: predictions are capped at the largest future_spend seen in TRAIN, so no absurd values.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent))
import config
from eda import save_matplotlib
from feature_selection import make_model_matrix, spend_target

MODEL_NAMES = ["Linear Regression", "Random Forest", "XGBoost"]

# Small grids ("light tuning"), only for Random Forest and XGBoost; tuned on the log target.
PARAM_GRIDS = {
    "Random Forest": {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [5, 20]},
    "XGBoost": {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1],
                "model__n_estimators": [200, 400]},
}


def load_spend_data():
    """Return X_train, y_train_log, X_test, y_test_gbp and the two tables, using the 8 selected spend features."""
    selected = json.loads(config.SELECTED_FEATURES_JSON.read_text())["spend"]
    encoder = joblib.load(config.SEGMENT_ENCODER)
    train = pd.read_parquet(config.TRAIN_TABLE_PARQUET)
    test = pd.read_parquet(config.TEST_TABLE_PARQUET)
    X_train = make_model_matrix(train, encoder)[selected]
    X_test = make_model_matrix(test, encoder)[selected]
    return X_train, spend_target(train), X_test, test["future_spend"], train, test


def make_pipeline(name):
    """Return an untrained regression pipeline (scaling only for the linear model)."""
    rs = config.RANDOM_STATE
    if name == "Linear Regression":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    if name == "Random Forest":
        return Pipeline([("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                         random_state=rs, n_jobs=1))])
    if name == "XGBoost":
        return Pipeline([("model", XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                subsample=0.8, colsample_bytree=0.8,
                                                random_state=rs, n_jobs=1))])
    raise ValueError(f"Unknown model {name}")


def cv_splitter():
    """Shuffled 5-fold split used for every cross-validation on TRAIN."""
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def tune(name, X_train, y_train_log):
    """Return an untrained pipeline with the best small-grid settings (found on the log target, TRAIN only)."""
    pipeline = make_pipeline(name)
    if name not in PARAM_GRIDS:
        return pipeline, {}
    search = GridSearchCV(pipeline, PARAM_GRIDS[name], scoring=config.SPEND_CV_SCORING, cv=cv_splitter(),
                          n_jobs=config.N_JOBS)
    search.fit(X_train, y_train_log)
    return pipeline.set_params(**search.best_params_), search.best_params_


def smearing_factor(model, X, y_log):
    """Duan's smearing factor S = mean(exp(residual)) on the log scale. Call with TRAIN data only."""
    residuals = np.asarray(y_log) - model.predict(X)
    return float(np.mean(np.exp(residuals)))


def predict_gbp(model, X, smearing=1.0, cap=None):
    """Predict spend in GBP: exp(log prediction) * smearing - 1, clipped to [0, cap].

    smearing = 1.0 gives the plain expm1 back-transform.
    """
    predicted = np.exp(model.predict(X)) * smearing - 1
    return np.clip(predicted, 0, cap)


def cross_validate_gbp(pipeline, X_train, y_train_log, use_smearing):
    """Out-of-fold CV on TRAIN: return CV RMSE on the log scale and GBP metrics of out-of-fold predictions.

    With use_smearing, each fold learns its own smearing factor from its own training part.
    """
    y_log = np.asarray(y_train_log)
    oof_gbp = np.zeros(len(y_log))
    log_rmse = []
    for train_idx, valid_idx in cv_splitter().split(X_train):
        model = clone(pipeline).fit(X_train.iloc[train_idx], y_log[train_idx])
        factor = smearing_factor(model, X_train.iloc[train_idx], y_log[train_idx]) if use_smearing else 1.0
        oof_gbp[valid_idx] = predict_gbp(model, X_train.iloc[valid_idx], factor)
        log_rmse.append(np.sqrt(mean_squared_error(y_log[valid_idx], model.predict(X_train.iloc[valid_idx]))))
    y_gbp = np.expm1(y_log)
    return {
        "cv_rmse_log": float(np.mean(log_rmse)),
        "cv_rmse_gbp": float(np.sqrt(mean_squared_error(y_gbp, oof_gbp))),
        "cv_mae_gbp": mean_absolute_error(y_gbp, oof_gbp),
        "cv_r2_gbp": r2_score(y_gbp, oof_gbp),
        "cv_mean_predicted": float(oof_gbp.mean()),
    }


def spend_metrics(y_true_gbp, predicted_gbp):
    """Return MAE, RMSE, R-squared in GBP, MAE for real spenders, R-squared on log scale, mean and max."""
    y_true_gbp = np.asarray(y_true_gbp)
    spenders = y_true_gbp > 0
    return {
        "mae": mean_absolute_error(y_true_gbp, predicted_gbp),
        "rmse": float(np.sqrt(mean_squared_error(y_true_gbp, predicted_gbp))),
        "r2": r2_score(y_true_gbp, predicted_gbp),
        "mae_spenders_only": mean_absolute_error(y_true_gbp[spenders], predicted_gbp[spenders]),
        "r2_log_scale": r2_score(np.log1p(y_true_gbp), np.log1p(predicted_gbp)),
        "mean_predicted": float(np.mean(predicted_gbp)),
        "mean_actual": float(np.mean(y_true_gbp)),
        "max_predicted": float(np.max(predicted_gbp)),
        "max_actual": float(np.max(y_true_gbp)),
    }


def historical_rate_baseline(table):
    """Past spend per 6 months: monetary / max(tenure_days, 30) * 182. Uses only pre-cutoff features."""
    tenure = np.maximum(table["tenure_days"], config.HIST_RATE_MIN_TENURE_DAYS)
    return (table["monetary"] / tenure * config.HIST_RATE_HORIZON_DAYS).to_numpy()


def baseline_cv_rows(train_df):
    """Score the two baselines on TRAIN (no fitting needed except the median, which is the same in every fold)."""
    y = train_df["future_spend"].to_numpy()
    rows = []
    for name, predicted in [("Historical-rate baseline", historical_rate_baseline(train_df)),
                            ("Median baseline (train median)", np.full(len(y), np.median(y)))]:
        rows.append({"model": name, "cv_rmse_log": np.nan,
                     "cv_rmse_gbp": float(np.sqrt(mean_squared_error(y, predicted))),
                     "cv_mae_gbp": mean_absolute_error(y, predicted), "cv_r2_gbp": r2_score(y, predicted),
                     "cv_mean_predicted": float(predicted.mean())})
    return rows


def run_train_cv(X_train, y_train_log, train_df):
    """Tune each model and score it with out-of-fold CV on TRAIN, without and with smearing.

    Returns (CV table, dict of tuned unfitted pipelines, dict of best params).
    """
    rows, pipelines, params = baseline_cv_rows(train_df), {}, {}
    for name in MODEL_NAMES:
        pipelines[name], params[name] = tune(name, X_train, y_train_log)
        for use_smearing in [False, True]:
            label = f"{name} + smearing" if use_smearing else name
            rows.append({"model": label, **cross_validate_gbp(pipelines[name], X_train, y_train_log, use_smearing)})
            print(f"CV done: {label}")
    return pd.DataFrame(rows), pipelines, params


def choose_final_model(cv_table):
    """Pick the model (plain or + smearing) with the lowest out-of-fold CV RMSE in GBP on TRAIN."""
    models_only = cv_table.dropna(subset=["cv_rmse_log"])
    return models_only.loc[models_only["cv_rmse_gbp"].idxmin(), "model"]


def fit_and_test(pipelines, X_train, y_train_log, X_test, y_test_gbp, train_df, test_df):
    """Fit every model on all of TRAIN and score it on TEST without/with smearing, plus both baselines.

    Predictions are capped at the TRAIN maximum future_spend; the number of capped values is reported.
    Returns (test table, fitted models, smearing factors, test predictions).
    """
    cap = float(train_df["future_spend"].max())
    predictions = {
        "Historical-rate baseline": historical_rate_baseline(test_df),
        "Median baseline (train median)": np.full(len(test_df), train_df["future_spend"].median()),
    }
    capped_counts = {name: 0 for name in predictions}
    fitted, factors = {}, {}
    for name, pipeline in pipelines.items():
        model = clone(pipeline).fit(X_train, y_train_log)
        fitted[name] = model
        factors[name] = smearing_factor(model, X_train, y_train_log)     # TRAIN only
        for label, factor in [(name, 1.0), (f"{name} + smearing", factors[name])]:
            uncapped = predict_gbp(model, X_test, factor)
            capped_counts[label] = int((uncapped > cap).sum())
            predictions[label] = np.minimum(uncapped, cap)
    rows = [{"model": label, "n_capped": capped_counts[label], **spend_metrics(y_test_gbp, pred)}
            for label, pred in predictions.items()]
    return pd.DataFrame(rows), fitted, factors, predictions, cap


def check_leakage_alarm(table):
    """Stop with an error if any row has a suspiciously high test R-squared (possible leakage)."""
    suspicious = table[table["r2"] > config.LEAKAGE_ALARM_R2]
    if len(suspicious) > 0:
        raise RuntimeError(f"Test R2 above {config.LEAKAGE_ALARM_R2} - check for leakage:\n{suspicious}")
    print(f"Leakage alarm check: no row above test R2 {config.LEAKAGE_ALARM_R2} (max {table['r2'].max():.3f})")


def plot_predicted_vs_actual(y_test_gbp, predicted_gbp, name, file_name="spend_predicted_vs_actual"):
    """Save a scatter of predicted vs actual future spend on log1p axes, with the perfect-prediction line."""
    actual = np.log1p(np.asarray(y_test_gbp))
    predicted = np.log1p(predicted_gbp)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, s=6, alpha=0.4)
    top = max(actual.max(), predicted.max())
    ax.plot([0, top], [0, top], "--", color="grey", label="perfect prediction")
    ticks = [0, 10, 100, 1000, 10000, 100000]
    ax.set_xticks(np.log1p(ticks), [f"{t:,}" for t in ticks])
    ax.set_yticks(np.log1p(ticks), [f"{t:,}" for t in ticks])
    ax.set_xlabel("Actual 6-month spend (GBP, log scale)")
    ax.set_ylabel("Predicted 6-month spend (GBP, log scale)")
    ax.set_title(f"Predicted vs actual on TEST - {name}")
    ax.legend()
    save_matplotlib(fig, file_name)
    return fig


def save_spend_model(model, name, features, smearing, cap):
    """Save the final spend pipeline with its smearing factor and cap (both learned on TRAIN)."""
    joblib.dump({"model": model, "name": name, "features": features, "target": "log1p(future_spend)",
                 "smearing_factor": smearing, "cap_gbp": cap}, config.SPEND_MODEL)
