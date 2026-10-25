"""Train everything from (raw or clean) data to the saved models the app uses.

Run from the project folder:   python src/train.py

Steps:
  1. clean data            (only if data/processed/transactions_clean.parquet is missing)
  2. train / test tables   (features before each cutoff, labels after it, active customers only)
  3. segments              (scaler + K-Means on TRAIN)                 -> models/segmentation.pkl
  4. feature selection     (filter, RFE, embedded, on TRAIN)           -> models/selected_features.json,
                                                                          models/segment_encoder.pkl
  5. churn model           (final choice from notebook 04, tuned on TRAIN) -> models/churn_model.pkl
  6. spend model           (two-part model, parts chosen by TRAIN CV)  -> models/spend_model.pkl
  7. decision thresholds   (risk/value splits + profit-rule settings)  -> models/decision_thresholds.json
  8. training medians      (used to fill values the form cannot ask)  -> models/training_medians.json

Model SELECTION happened in the notebooks (04, 05); this script refits those choices from scratch.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
import churn_models
import config
import decision_rules
import hurdle_model
import segmentation
import spend_models
from data_cleaning import run_cleaning
from feature_builder import FEATURE_COLUMNS
from feature_selection import run_feature_selection
from snapshots import build_train_test_tables


def step(message):
    """Print a step header with the time, so progress is easy to follow."""
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}")


def save_training_medians(train_df):
    """Save TRAIN medians used to fill form fields that cannot be typed.

    Some medians are taken from the matching group only (e.g. purchase_regularity only from customers with
    3+ orders), because for other customers the value is fixed by a rule, not measured.
    """
    trend_days = config.TREND_WINDOW_MONTHS * config.DAYS_PER_MONTH
    mixed_trend = train_df[(train_df["recency"] <= trend_days) & (train_df["tenure_days"] > trend_days)]
    medians = {column: float(train_df[column].median()) for column in FEATURE_COLUMNS}
    medians["purchase_regularity_3plus_orders"] = float(
        train_df.loc[train_df["frequency"] >= 3, "purchase_regularity"].median())
    medians["spend_trend_mixed_history"] = float(mixed_trend["spend_trend"].median())
    medians["tenure_per_order_days"] = float((train_df["tenure_days"] / train_df["frequency"]).median())
    config.TRAINING_MEDIANS_JSON.write_text(json.dumps(medians, indent=2))
    return medians


def train_spend_model(train_df):
    """Choose the two parts by TRAIN CV, fit the two-part spend model on all of TRAIN and save it."""
    X_train, _, _, _, _, _ = spend_models.load_spend_data()
    y = train_df["future_spend"].to_numpy()
    spenders = y > 0
    part1_table, classifiers = hurdle_model.evaluate_part1(X_train, spenders.astype(int))
    part2_table, regressors = hurdle_model.evaluate_part2(X_train[spenders], np.log1p(y[spenders]))
    part1, part2 = hurdle_model.choose_parts(part1_table, part2_table)
    bundle = hurdle_model.fit_hurdle(classifiers[part1], regressors[part2], X_train, y)
    cap = float(train_df["future_spend"].max())
    hurdle_model.save_hurdle_model(bundle, list(X_train.columns), cap, f"{part1} + {part2}")
    print(f"  parts: {part1} + {part2}, smearing factor {bundle['smearing_factor']:.3f}, cap {cap:,.0f} GBP")


def sanity_check():
    """Score the saved models on TEST, so a broken run is noticed immediately."""
    test = pd.read_parquet(config.TEST_TABLE_PARQUET)
    scores = decision_rules.score_customers(test)
    print(f"  churn ROC-AUC on TEST:        {roc_auc_score(test['churn'], scores['churn_probability']):.4f}")
    print(f"  spend R-squared on TEST (GBP): {r2_score(test['future_spend'], scores['expected_spend']):.4f}")
    print(f"  customers passing profit rule: {(scores['expected_net'] > 0).sum():,} of {len(scores):,}")


def main():
    """Run the full training pipeline."""
    start = time.time()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    step("1/8 Clean data")
    if config.TRANSACTIONS_CLEAN_PARQUET.exists():
        print("  clean data found, skipping")
    else:
        run_cleaning()

    step("2/8 Build train and test tables")
    train, test = build_train_test_tables()

    step("3/8 Fit segments (TRAIN only)")
    bundle = segmentation.fit_segmentation(train, config.N_SEGMENTS)
    train, test = segmentation.add_segments_to_tables(train, test, bundle)
    print(f"  segments: {bundle['segment_names']}")

    step("4/8 Feature selection (TRAIN only)")
    selected = run_feature_selection(train)
    print(f"  churn features: {selected['churn']}")
    print(f"  spend features: {selected['spend']}")

    step(f"5/8 Churn model: {config.FINAL_CHURN_MODEL} | {config.FINAL_CHURN_IMBALANCE} | monotonic")
    X_train, y_train, _, _, _, _ = churn_models.load_churn_data()
    model, cv_score, params = churn_models.fit_with_cv(config.FINAL_CHURN_MODEL, config.FINAL_CHURN_IMBALANCE,
                                                       X_train, y_train, monotone=config.CHURN_MONOTONE_CONSTRAINTS)
    churn_models.save_churn_model(model, f"{config.FINAL_CHURN_MODEL} | {config.FINAL_CHURN_IMBALANCE} | monotonic",
                                  list(X_train.columns))
    print(f"  CV ROC-AUC on TRAIN {cv_score:.4f}, params {params}")

    step("6/8 Spend model (two-part)")
    train_spend_model(train)

    step("7/8 Decision thresholds (TRAIN only)")
    thresholds = decision_rules.fit_split_points(decision_rules.score_customers(train))
    print(f"  risk split {thresholds['risk_threshold']:.3f}, value split {thresholds['value_threshold']:,.1f} GBP")

    step("8/8 Training medians for the app")
    save_training_medians(train)

    step("Sanity check on TEST")
    sanity_check()
    print(f"\nDone in {time.time() - start:.0f} s. Saved files:")
    for path in sorted(config.MODELS_DIR.iterdir()):
        print(f"  {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
