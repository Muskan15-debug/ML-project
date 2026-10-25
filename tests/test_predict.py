"""Tests for the prediction functions used by the app (form and CSV upload)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import config
import decision_rules
from predict import WIN_BACK_WARNING, predict_batch, predict_customer

SPEC_EXAMPLE = {"recency": 85, "frequency": 3, "monetary": 240, "avg_gap_days": 40, "tenure_days": 200}
FORM_FIELDS = ["recency", "frequency", "monetary", "tenure_days", "unique_products", "return_rate", "is_uk"]
ALL_FIELDS = FORM_FIELDS + ["spend_trend", "purchase_regularity"]


@pytest.fixture(scope="module")
def test_table():
    """The saved TEST snapshot table."""
    return pd.read_parquet(config.TEST_TABLE_PARQUET)


def check_output(result):
    """Check that one predict_customer result has the right keys, types and ranges."""
    assert isinstance(result["verdict"], str) and result["verdict"].startswith(("Likely to leave", "Likely to stay"))
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert result["expected_spend_6m"] >= 0
    assert result["conditional_amount_6m"] >= 0
    assert result["expected_spend_6m"] <= result["conditional_amount_6m"] + 1e-6   # P(spend>0) <= 1
    assert isinstance(result["expected_net"], float)
    assert result["segment"] in {"Champions", "Loyal", "One-off buyers", "At-Risk"}
    assert 1 <= len(result["top_reasons"]) <= 5 and all(isinstance(r, str) for r in result["top_reasons"])
    assert isinstance(result["action"], str) and isinstance(result["action_reason"], str)
    assert result["features_used"]["avg_order_value"] == pytest.approx(
        result["features_used"]["monetary"] / result["features_used"]["frequency"])


def test_spec_example():
    """The spec's example customer gives a complete, valid result and reports what it assumed."""
    result = predict_customer(SPEC_EXAMPLE)
    check_output(result)
    assert "avg_order_value" in result["derived_values"]
    assert "purchase_regularity" in result["assumed_values"]       # cannot be typed for 3+ orders
    # Typed gap 40 does not match (200 - 85) / 2 = 57.5, so a warning is expected.
    assert any("average gap" in w for w in result["warnings"])
    assert result["features_used"]["avg_gap_days"] == pytest.approx(57.5)


@pytest.mark.parametrize("position", [0, 1000, 2000])
def test_customers_from_test_table(test_table, position):
    """Three real test customers typed in through the form fields give valid results."""
    row = test_table.iloc[position]
    result = predict_customer({field: row[field] for field in FORM_FIELDS})
    check_output(result)


def test_same_pipeline_as_training(test_table):
    """With ALL features given, the app's probabilities equal the saved model's scores on the test table."""
    sample = test_table.iloc[:50]
    batch = predict_batch(sample[ALL_FIELDS].reset_index())
    expected = decision_rules.score_customers(sample)["churn_probability"].round(3)
    got = batch.set_index("customer_id")["churn_probability"]
    np.testing.assert_allclose(got.loc[expected.index].to_numpy(), expected.to_numpy(), atol=1e-3)


def test_win_back_warning():
    """Recency above the active window still returns a result, with the win-back warning."""
    result = predict_customer({"recency": 250, "frequency": 2, "monetary": 300, "tenure_days": 400})
    check_output(result)
    assert WIN_BACK_WARNING in result["warnings"]


def test_single_order_customer():
    """One-order customers get gap = recency and regularity = 0, as in training."""
    result = predict_customer({"recency": 20, "frequency": 1, "monetary": 50})
    check_output(result)
    assert result["features_used"]["avg_gap_days"] == 20
    assert result["features_used"]["purchase_regularity"] == 0
    assert result["features_used"]["tenure_days"] == 20


@pytest.mark.parametrize("bad_input", [
    {"recency": 10, "frequency": 0, "monetary": 100},
    {"recency": -1, "frequency": 2, "monetary": 100},
    {"recency": 10, "frequency": 2, "monetary": 0},
    {"recency": 100, "frequency": 2, "monetary": 100, "tenure_days": 50},
    {"frequency": 2, "monetary": 100},
])
def test_invalid_inputs_raise(bad_input):
    """Impossible inputs are rejected with a clear ValueError."""
    with pytest.raises(ValueError):
        predict_customer(bad_input)


def test_batch_ranked_and_errors_kept(test_table):
    """Batch output is ranked by expected net profit, and bad rows are reported, not dropped silently."""
    good = test_table.iloc[:5][FORM_FIELDS].reset_index()
    bad = pd.DataFrame([{"customer_id": "bad", "recency": 5, "frequency": 0, "monetary": 10}])
    result = predict_batch(pd.concat([good, bad], ignore_index=True))
    scored = result.dropna(subset=["rank"])
    assert len(scored) == 5
    assert scored["expected_net"].is_monotonic_decreasing
    assert scored["churn_probability"].between(0, 1).all()
    assert (scored["expected_spend_6m"] >= 0).all()
    assert result.loc[result["customer_id"] == "bad", "error"].notna().all()
