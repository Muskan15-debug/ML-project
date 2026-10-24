"""Combine features and labels into a snapshot table for one cutoff, and compare cutoffs."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
import config
from feature_builder import build_features
from labels import build_labels, label_window_ends


def load_clean_data():
    """Load the cleaned transactions and the cancellations table saved in Phase 1."""
    transactions = pd.read_parquet(config.TRANSACTIONS_CLEAN_PARQUET)
    cancellations = pd.read_parquet(config.CANCELLATIONS_PARQUET)
    return transactions, cancellations


def keep_active_customers(table):
    """Keep only customers who bought within ACTIVE_WINDOW_DAYS before the cutoff (recency <= window)."""
    return table[table["recency"] <= config.ACTIVE_WINDOW_DAYS]


def build_snapshot(transactions, cancellations, cutoff_date):
    """Return features (before cutoff) joined with labels (after cutoff), for ACTIVE customers only."""
    features = build_features(transactions, cancellations, cutoff_date)
    labels = build_labels(transactions, cutoff_date)
    snapshot = features.join(labels, how="inner")
    # Population filter, not a feature change: features were built from the full history above.
    return keep_active_customers(snapshot)


def check_data_covers(transactions, cutoff_date):
    """Raise an error if the data ends before the label windows of this cutoff end."""
    _, spend_end = label_window_ends(cutoff_date)
    last_date = transactions["invoice_date"].max()
    if last_date < spend_end - pd.Timedelta(days=1):
        raise ValueError(f"Data ends {last_date}, but spend window of {cutoff_date.date()} "
                         f"needs data until {spend_end.date()}")


def recency_baseline(snapshot):
    """Score the simple rule 'churn if recency > median recency' and recency as a ranking score.

    This shows how much churn is explained by recency alone, before any model.
    """
    predicted = (snapshot["recency"] > snapshot["recency"].median()).astype(int)
    y = snapshot["churn"]
    return {
        "median_recency": round(snapshot["recency"].median(), 1),
        "baseline_precision": round(precision_score(y, predicted), 3),
        "baseline_recall": round(recall_score(y, predicted), 3),
        "baseline_f1": round(f1_score(y, predicted), 3),
        "recency_roc_auc": round(roc_auc_score(y, snapshot["recency"]), 3),
    }


def compare_cutoffs(transactions, cancellations, train_candidates, test_cutoff):
    """Return a small table describing each candidate train cutoff and the test cutoff (active customers)."""
    first_date = transactions["invoice_date"].min()
    rows = []
    for role, cutoff in [("train?", c) for c in train_candidates] + [("test", test_cutoff)]:
        check_data_covers(transactions, cutoff)
        churn_end, spend_end = label_window_ends(cutoff)
        snapshot = build_snapshot(transactions, cancellations, cutoff)
        row = {
            "role": role,
            "cutoff": cutoff.date(),
            "ends_before_test": role == "test" or max(churn_end, spend_end) <= test_cutoff,
            "history_months": round((cutoff - first_date) / pd.Timedelta(days=30.44), 1),
            "customers": len(snapshot),
            "churn_rate": round(snapshot["churn"].mean(), 3),
            "spend_mean": round(snapshot["future_spend"].mean(), 1),
            "spend_median": round(snapshot["future_spend"].median(), 1),
            "spend_zero_share": round((snapshot["future_spend"] == 0).mean(), 3),
        }
        row.update(recency_baseline(snapshot))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_snapshot(name, table):
    """Print customers, churn rate and a future_spend summary for one snapshot table."""
    spend = table["future_spend"]
    print(f"--- {name} ---")
    print(f"Customers:                 {len(table):,}")
    print(f"Churn rate:                {table['churn'].mean():.3f}  "
          f"(churn=1: {int(table['churn'].sum()):,}, churn=0: {int((table['churn'] == 0).sum()):,})")
    print(f"future_spend = 0:          {(spend == 0).sum():,} customers ({(spend == 0).mean():.1%})")
    print("future_spend summary (GBP):")
    print(spend.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).round(1).to_string())
    print()


def build_train_test_tables():
    """Build and save the out-of-time train (TRAIN_CUTOFF) and test (TEST_CUTOFF) tables."""
    transactions, cancellations = load_clean_data()
    tables = {}
    for name, cutoff, path in [("TRAIN", config.TRAIN_CUTOFF, config.TRAIN_TABLE_PARQUET),
                               ("TEST", config.TEST_CUTOFF, config.TEST_TABLE_PARQUET)]:
        check_data_covers(transactions, cutoff)
        table = build_snapshot(transactions, cancellations, cutoff)
        table.to_parquet(path)
        print(f"Saved {path.name} (cutoff {cutoff.date()})")
        tables[name] = table
    print()
    for name, table in tables.items():
        summarise_snapshot(name, table)
    return tables["TRAIN"], tables["TEST"]


if __name__ == "__main__":
    build_train_test_tables()
