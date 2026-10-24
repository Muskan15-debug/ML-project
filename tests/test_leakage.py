"""Tests that features never see the future and labels never overlap the feature period."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import config
from feature_builder import build_features
from labels import build_labels, label_window_ends
from snapshots import build_snapshot, load_clean_data

ALL_CUTOFFS = config.CANDIDATE_TRAIN_CUTOFFS + [config.TRAIN_CUTOFF, config.TEST_CUTOFF]


@pytest.fixture(scope="module")
def data():
    """Load the real cleaned data once for all tests."""
    return load_clean_data()


def make_poisoned_future(transactions, cancellations, as_of_date):
    """Return copies of the data with huge fake rows added on and after as_of_date.

    If build_features used any row with invoice_date >= as_of_date, these rows
    would change the features, so comparing before/after detects leakage.
    """
    customers = transactions.loc[transactions["invoice_date"] < as_of_date, "customer_id"].unique()[:200]
    fake = pd.DataFrame({
        "invoice": [f"FUTURE{i}" for i in range(len(customers))],
        "stock_code": "FAKE",
        "description": "FAKE",
        "quantity": 1000,
        "invoice_date": as_of_date,          # exactly ON the cutoff: must be excluded too
        "price": 999.0,
        "customer_id": customers,
        "country": "Fakeland",
        "total_price": 999000.0,
    })
    poisoned_tx = pd.concat([transactions, fake], ignore_index=True)

    fake_cancels = pd.DataFrame({
        "invoice": [f"CFUTURE{i}" for i in range(len(customers))],
        "invoice_date": as_of_date + pd.Timedelta(days=1),
        "customer_id": customers,
        "stock_code": "FAKE",
        "quantity": -1,
        "price": 1.0,
        "cancelled_value": 1.0,
    })
    poisoned_cancels = pd.concat([cancellations, fake_cancels], ignore_index=True)
    return poisoned_tx, poisoned_cancels


@pytest.mark.parametrize("as_of_date", [config.TRAIN_CUTOFF, config.TEST_CUTOFF])
def test_features_ignore_rows_on_or_after_as_of_date(data, as_of_date):
    """Adding fake future rows must not change any feature value."""
    transactions, cancellations = data
    original = build_features(transactions, cancellations, as_of_date)
    poisoned_tx, poisoned_cancels = make_poisoned_future(transactions, cancellations, as_of_date)
    poisoned = build_features(poisoned_tx, poisoned_cancels, as_of_date)
    pd.testing.assert_frame_equal(original, poisoned)


@pytest.mark.parametrize("as_of_date", [config.TRAIN_CUTOFF, config.TEST_CUTOFF])
def test_features_equal_when_future_rows_removed(data, as_of_date):
    """Features from all data must equal features from data already cut at as_of_date."""
    transactions, cancellations = data
    full = build_features(transactions, cancellations, as_of_date)
    cut = build_features(transactions[transactions["invoice_date"] < as_of_date],
                         cancellations[cancellations["invoice_date"] < as_of_date], as_of_date)
    pd.testing.assert_frame_equal(full, cut)


def test_train_label_windows_end_before_test_cutoff():
    """The train labels must come from a period that ends before the test snapshot starts."""
    churn_end, spend_end = label_window_ends(config.TRAIN_CUTOFF)
    assert churn_end <= config.TEST_CUTOFF
    assert spend_end <= config.TEST_CUTOFF


def test_data_covers_test_spend_window(data):
    """The dataset must reach (almost) the end of the test spend window."""
    transactions, _ = data
    _, spend_end = label_window_ends(config.TEST_CUTOFF)
    assert transactions["invoice_date"].max() >= spend_end - pd.Timedelta(days=1)


@pytest.mark.parametrize("as_of_date", ALL_CUTOFFS)
def test_no_negative_recency_or_tenure(data, as_of_date):
    """Recency and tenure are days before the cutoff, so they can never be negative."""
    transactions, cancellations = data
    features = build_features(transactions, cancellations, as_of_date)
    assert (features["recency"] >= 0).all()
    assert (features["tenure_days"] >= 0).all()
    assert (features["tenure_days"] >= features["recency"]).all()


@pytest.mark.parametrize("cutoff", [config.TRAIN_CUTOFF, config.TEST_CUTOFF])
def test_labels_only_for_customers_seen_before_cutoff(data, cutoff):
    """Every labelled customer must have bought before the cutoff (so features exist)."""
    transactions, cancellations = data
    labels = build_labels(transactions, cutoff)
    features = build_features(transactions, cancellations, cutoff)
    assert set(labels.index) == set(features.index)
    assert set(labels["churn"].unique()) <= {0, 1}
    assert (labels["future_spend"] >= 0).all()


@pytest.mark.parametrize("cutoff", [config.TRAIN_CUTOFF, config.TEST_CUTOFF])
def test_snapshot_contains_only_active_customers(data, cutoff):
    """Every customer in a snapshot must have bought within ACTIVE_WINDOW_DAYS before the cutoff."""
    transactions, cancellations = data
    snapshot = build_snapshot(transactions, cancellations, cutoff)
    assert len(snapshot) > 0
    assert (snapshot["recency"] <= config.ACTIVE_WINDOW_DAYS).all()


@pytest.mark.parametrize("path", [config.TRAIN_TABLE_PARQUET, config.TEST_TABLE_PARQUET])
def test_saved_tables_contain_only_active_customers(path):
    """The saved train/test tables must also respect the active-customer rule."""
    if not path.exists():
        pytest.skip(f"{path.name} not built yet")
    table = pd.read_parquet(path)
    assert (table["recency"] <= config.ACTIVE_WINDOW_DAYS).all()
