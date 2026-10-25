"""Tests for the prediction functions used by the app (form and CSV upload)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import config
import decision_rules
from predict import WIN_BACK_WARNING, predict_batch, predict_customer, table_to_inputs

# Spec section 9 example, completed with the extra fields the form now asks for.
SPEC_EXAMPLE = {"recency": 85, "frequency": 3, "monetary": 240, "tenure_days": 200, "unique_products": 6,
                "spend_last_3m": 80, "returned_orders": 0, "is_uk": 1}


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
    used = result["features_used"]
    assert used["avg_order_value"] == pytest.approx(used["monetary"] / used["frequency"])
    # Only purchase_regularity may ever be assumed.
    assert set(result["assumed_values"]) <= {"purchase_regularity"}


def test_spec_example():
    """The spec's example customer gives a complete, valid result with calculated values."""
    result = predict_customer(SPEC_EXAMPLE)
    check_output(result)
    used = result["features_used"]
    assert used["avg_gap_days"] == pytest.approx((200 - 85) / 2)
    assert used["spend_trend"] == pytest.approx(80 / 240)
    assert used["return_rate"] == 0
    assert "purchase_regularity" in result["assumed_values"]       # 3 orders: cannot be typed


@pytest.mark.parametrize("position", [0, 1000, 2000])
def test_customers_from_test_table(test_table, position):
    """Three real test customers typed in through the form fields give valid results."""
    inputs = table_to_inputs(test_table.iloc[[position]]).iloc[0].to_dict()
    check_output(predict_customer(inputs))


def test_same_pipeline_as_training(test_table):
    """With all values given, the app's probabilities equal the saved model's scores on the test table."""
    sample = test_table.iloc[:50]
    batch = predict_batch(table_to_inputs(sample, include_regularity=True))
    expected = decision_rules.score_customers(sample)["churn_probability"].round(3)
    got = batch.set_index("customer_id")["churn_probability"]
    np.testing.assert_allclose(got.loc[expected.index].to_numpy(), expected.to_numpy(), atol=2e-3)


def test_win_back_warning():
    """Recency above the active window still returns a result, with the win-back warning."""
    result = predict_customer({**SPEC_EXAMPLE, "recency": 250, "tenure_days": 400, "spend_last_3m": 0})
    check_output(result)
    assert WIN_BACK_WARNING in result["warnings"]


def test_single_order_customer():
    """One-order customers: tenure = recency, gap = recency, regularity = 0 (training rules), no assumptions."""
    result = predict_customer({"recency": 20, "frequency": 1, "monetary": 50, "unique_products": 3,
                               "spend_last_3m": 50, "returned_orders": 0, "is_uk": "yes"})
    check_output(result)
    used = result["features_used"]
    assert used["tenure_days"] == 20 and used["avg_gap_days"] == 20 and used["purchase_regularity"] == 0
    assert result["assumed_values"] == {}


@pytest.mark.parametrize("change", [
    {"frequency": 0},                       # no orders
    {"recency": -1},                        # negative days
    {"monetary": 0},                        # nothing spent
    {"unique_products": 0},                 # no products
    {"returned_orders": -1},                # negative returns
    {"spend_last_3m": 300},                 # more than total spent
    {"tenure_days": 50},                    # first order after last order
    {"recency": 120, "spend_last_3m": 10},  # last order > 3 months ago but recent spend
    {"is_uk": "maybe"},                     # not yes/no
    {"monetary": None},                     # missing value
])
def test_invalid_inputs_raise(change):
    """Impossible or missing inputs are rejected with a clear ValueError."""
    with pytest.raises(ValueError):
        predict_customer({**SPEC_EXAMPLE, **change})


def test_batch_ranked_and_errors_kept(test_table):
    """Batch output is ranked by expected net profit, and bad rows are reported, not dropped silently."""
    good = table_to_inputs(test_table.iloc[:5])
    bad = pd.DataFrame([{**SPEC_EXAMPLE, "customer_id": "bad", "frequency": 0}])
    result = predict_batch(pd.concat([good, bad], ignore_index=True))
    scored = result.dropna(subset=["rank"])
    assert len(scored) == 5
    assert scored["expected_net"].is_monotonic_decreasing
    assert scored["churn_probability"].between(0, 1).all()
    assert (scored["expected_spend_6m"] >= 0).all()
    assert result.loc[result["customer_id"] == "bad", "error"].notna().all()
