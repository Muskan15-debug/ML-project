"""Combine features and labels into a snapshot table for one cutoff, and compare cutoffs."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config
from feature_builder import build_features
from labels import build_labels, label_window_ends


def load_clean_data():
    """Load the cleaned transactions and the cancellations table saved in Phase 1."""
    transactions = pd.read_parquet(config.TRANSACTIONS_CLEAN_PARQUET)
    cancellations = pd.read_parquet(config.CANCELLATIONS_PARQUET)
    return transactions, cancellations


def build_snapshot(transactions, cancellations, cutoff_date):
    """Return features (before cutoff) joined with labels (after cutoff), one row per customer."""
    features = build_features(transactions, cancellations, cutoff_date)
    labels = build_labels(transactions, cutoff_date)
    return features.join(labels, how="inner")


def check_data_covers(transactions, cutoff_date):
    """Raise an error if the data ends before the label windows of this cutoff end."""
    _, spend_end = label_window_ends(cutoff_date)
    last_date = transactions["invoice_date"].max()
    if last_date < spend_end - pd.Timedelta(days=1):
        raise ValueError(f"Data ends {last_date}, but spend window of {cutoff_date.date()} "
                         f"needs data until {spend_end.date()}")


def compare_cutoffs(transactions, cancellations, train_candidates, test_cutoff):
    """Return a small table describing each candidate train cutoff and the test cutoff."""
    first_date = transactions["invoice_date"].min()
    rows = []
    for role, cutoff in [("train?", c) for c in train_candidates] + [("test", test_cutoff)]:
        check_data_covers(transactions, cutoff)
        churn_end, spend_end = label_window_ends(cutoff)
        labels = build_labels(transactions, cutoff)
        rows.append({
            "role": role,
            "cutoff": cutoff.date(),
            "label_windows_end": max(churn_end, spend_end).date(),
            "ends_before_test_cutoff": role == "test" or max(churn_end, spend_end) <= test_cutoff,
            "history_months": round((cutoff - first_date) / pd.Timedelta(days=30.44), 1),
            "customers": len(labels),
            "churn_rate": round(labels["churn"].mean(), 3),
            "future_spend_mean": round(labels["future_spend"].mean(), 1),
            "future_spend_median": round(labels["future_spend"].median(), 1),
            "future_spend_zero_share": round((labels["future_spend"] == 0).mean(), 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tx, cancels = load_clean_data()
    table = compare_cutoffs(tx, cancels, config.CANDIDATE_TRAIN_CUTOFFS, config.TEST_CUTOFF)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLES_DIR / "cutoff_comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
