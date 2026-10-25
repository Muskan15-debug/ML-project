"""Predict churn, value, segment and action for typed-in customers (form) or a CSV (batch).

Uses ONLY the saved files written by train.py:
  segmentation.pkl, segment_encoder.pkl, selected_features.json, churn_model.pkl,
  spend_model.pkl (two-part model, via hurdle_model.predict_hurdle), decision_thresholds.json,
  training_medians.json.

Inputs (column / key names):
  required: recency (days since last order), frequency (number of orders), monetary (total spent, GBP)
  optional: tenure_days, avg_gap_days, unique_products, return_rate, is_uk,
            and (advanced) spend_trend, purchase_regularity
  avg_order_value is ALWAYS calculated as monetary / frequency, never typed.

Every value that was not typed is either DERIVED (calculated exactly with the same rule as training)
or ASSUMED (filled with a TRAINING median). Both are reported back.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config
import decision_rules
from explain import explain_table, load_churn_parts
from feature_builder import FEATURE_COLUMNS
from segmentation import assign_segment, load_segmentation

REQUIRED_FIELDS = ["recency", "frequency", "monetary"]
OPTIONAL_FIELDS = ["tenure_days", "avg_gap_days", "unique_products", "return_rate", "is_uk",
                   "spend_trend", "purchase_regularity"]
WIN_BACK_WARNING = ("outside the training population - this is a win-back case, prediction is unreliable "
                    f"(the models were trained only on customers who bought in the last {config.ACTIVE_WINDOW_DAYS} days)")


@lru_cache(maxsize=1)
def load_artifacts():
    """Load every saved model file once and keep it in memory."""
    churn_bundle, encoder = load_churn_parts()
    return {
        "segmentation": load_segmentation(),
        "churn": churn_bundle,
        "encoder": encoder,
        "thresholds": decision_rules.load_split_points(),
        "medians": json.loads(config.TRAINING_MEDIANS_JSON.read_text()),
    }


def _given(inputs, key):
    """True if the input contains a real (non-empty) value for key."""
    value = inputs.get(key)
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and value != ""


def build_feature_row(inputs, medians):
    """Turn one input dict into the 12 model features.

    Returns (features dict, filled dict {feature: (value, 'derived' or 'assumed', explanation)}, warnings list).
    Raises ValueError for impossible inputs.
    """
    for key in REQUIRED_FIELDS:
        if not _given(inputs, key):
            raise ValueError(f"'{key}' is required")
    recency, frequency, monetary = float(inputs["recency"]), float(inputs["frequency"]), float(inputs["monetary"])
    if recency < 0:
        raise ValueError("recency (days since last order) cannot be negative")
    if frequency < 1 or frequency != int(frequency):
        raise ValueError("frequency (number of orders) must be a whole number >= 1")
    if monetary <= 0:
        raise ValueError("monetary (total spent) must be above 0")

    features = {"recency": recency, "frequency": frequency, "monetary": monetary}
    filled, warnings = {}, []
    trend_days = config.TREND_WINDOW_MONTHS * config.DAYS_PER_MONTH

    def fill(name, value, kind, why):
        features[name] = value
        filled[name] = (value, kind, why)

    # Average order value is always calculated, never typed (agreed rule).
    fill("avg_order_value", monetary / frequency, "derived", "total spent / number of orders")
    fill("is_single_order", int(frequency == 1), "derived", "1 if only one order")

    # Tenure (days since first order).
    if _given(inputs, "tenure_days"):
        features["tenure_days"] = float(inputs["tenure_days"])
    elif frequency == 1:
        fill("tenure_days", recency, "derived", "with one order, first order = last order")
    elif _given(inputs, "avg_gap_days"):
        fill("tenure_days", recency + float(inputs["avg_gap_days"]) * (frequency - 1), "derived",
             "recency + average gap x (orders - 1)")
    else:
        fill("tenure_days", max(recency, medians["tenure_per_order_days"] * frequency), "assumed",
             "training median days per order x number of orders")
    if features["tenure_days"] < recency:
        raise ValueError("days since first order (tenure) cannot be smaller than days since last order (recency)")
    if frequency > 1 and features["tenure_days"] == recency:
        raise ValueError("with more than one order, the first order must be earlier than the last order")

    # Average gap between orders: in training it is exactly (tenure - recency) / (orders - 1).
    if frequency == 1:
        fill("avg_gap_days", recency, "derived", "one order: gap so far = days since that order (training rule)")
    else:
        gap = (features["tenure_days"] - recency) / (frequency - 1)
        if _given(inputs, "avg_gap_days") and abs(float(inputs["avg_gap_days"]) - gap) > 1:
            warnings.append(f"typed average gap {float(inputs['avg_gap_days']):.1f} days does not match "
                            f"(tenure - recency) / (orders - 1) = {gap:.1f} days; using {gap:.1f} (training formula)")
        fill("avg_gap_days", gap, "derived", "(tenure - recency) / (orders - 1)")

    # Spread of the gaps between orders.
    if _given(inputs, "purchase_regularity"):
        features["purchase_regularity"] = float(inputs["purchase_regularity"])
    elif frequency <= 2:
        fill("purchase_regularity", 0.0, "derived", "fewer than 3 orders: no spread can be measured (training rule)")
    else:
        fill("purchase_regularity", medians["purchase_regularity_3plus_orders"], "assumed",
             "training median for customers with 3+ orders")

    # Share of spend in the last 3 months.
    if _given(inputs, "spend_trend"):
        features["spend_trend"] = float(inputs["spend_trend"])
    elif recency > trend_days:
        fill("spend_trend", 0.0, "derived", "no order in the last 3 months")
    elif features["tenure_days"] <= trend_days:
        fill("spend_trend", 1.0, "derived", "all orders are within the last 3 months")
    else:
        fill("spend_trend", medians["spend_trend_mixed_history"], "assumed",
             "training median for customers with orders both before and inside the last 3 months")

    for name, default_kind, why in [("unique_products", "assumed", "training median"),
                                    ("return_rate", "assumed", "training median"),
                                    ("is_uk", "assumed", "training median (most customers are UK)")]:
        if _given(inputs, name):
            features[name] = float(inputs[name])
        else:
            fill(name, medians[name], default_kind, why)

    if recency > config.ACTIVE_WINDOW_DAYS:
        warnings.append(WIN_BACK_WARNING)
    if frequency == 1:
        warnings.append("only one order: several values are set by rule, so confidence is lower")
    return {name: features[name] for name in FEATURE_COLUMNS}, filled, warnings


def risk_level(probability, thresholds):
    """Low below 50%, High at or above the TRAIN median churn probability, Medium in between."""
    if probability >= thresholds["risk_threshold"]:
        return "High"
    if probability >= config.RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def _score(feature_table):
    """Score a table of features: segment, churn, spend, expected_net, action and SHAP reasons."""
    art = load_artifacts()
    table = feature_table.join(assign_segment(feature_table, art["segmentation"]))
    scores = decision_rules.add_actions(decision_rules.score_customers(table), art["thresholds"])
    _, reasons = explain_table(table, art["churn"], art["encoder"])
    scores["top_reasons"] = reasons
    scores["risk_level"] = [risk_level(p, art["thresholds"]) for p in scores["churn_probability"]]
    high_value = scores["expected_spend"] >= art["thresholds"]["value_threshold"]
    scores["verdict"] = [("Likely to leave" if p >= config.VERDICT_THRESHOLD else "Likely to stay")
                         + (", high value" if hv else ", low value")
                         for p, hv in zip(scores["churn_probability"], high_value)]
    return scores


def predict_customer(input_dict):
    """Predict for ONE typed-in customer. Returns a dict with verdict, probability, risk, spend, segment,
    reasons, action, and which values were derived or assumed, plus warnings."""
    medians = load_artifacts()["medians"]
    features, filled, warnings = build_feature_row(input_dict, medians)
    scores = _score(pd.DataFrame([features], index=pd.Index(["customer"], name="customer_id")))
    r = scores.iloc[0]
    return {
        "verdict": r["verdict"],
        "churn_probability": float(r["churn_probability"]),
        "risk_level": r["risk_level"],
        "expected_spend_6m": float(r["expected_spend"]),
        "conditional_amount_6m": float(r["conditional_amount"]),
        "expected_net": float(r["expected_net"]),
        "segment": r["segment_name"],
        "top_reasons": list(r["top_reasons"]),
        "action": r["action"],
        "action_reason": r["reason"],
        "features_used": {k: float(v) for k, v in features.items()},
        "derived_values": {k: {"value": float(v), "how": why} for k, (v, kind, why) in filled.items() if kind == "derived"},
        "assumed_values": {k: {"value": float(v), "how": why} for k, (v, kind, why) in filled.items() if kind == "assumed"},
        "warnings": warnings,
    }


def predict_batch(df):
    """Predict for many customers (e.g. an uploaded CSV). Returns a table ranked by expected net profit.

    Rows with impossible inputs are kept with an 'error' message instead of stopping the whole batch.
    """
    medians = load_artifacts()["medians"]
    ids = df["customer_id"] if "customer_id" in df.columns else pd.Series(range(1, len(df) + 1), index=df.index)
    rows, info, errors = [], [], []
    for (_, record), customer_id in zip(df.iterrows(), ids):
        try:
            features, filled, warnings = build_feature_row(record.to_dict(), medians)
            rows.append(pd.Series(features, name=customer_id))
            info.append((customer_id, [k for k, (_, kind, _) in filled.items() if kind == "assumed"], warnings))
        except ValueError as error:
            errors.append({"customer_id": customer_id, "error": str(error)})

    result = pd.DataFrame()
    if rows:
        feature_table = pd.DataFrame(rows)
        feature_table.index.name = "customer_id"
        scores = _score(feature_table)
        result = pd.DataFrame({
            "customer_id": scores.index,
            "verdict": scores["verdict"].values,
            "churn_probability": scores["churn_probability"].round(3).values,
            "risk_level": scores["risk_level"].values,
            "expected_spend_6m": scores["expected_spend"].round(0).values,
            "conditional_amount_6m": scores["conditional_amount"].round(0).values,
            "expected_net": scores["expected_net"].round(1).values,
            "segment": scores["segment_name"].values,
            "action": scores["action"].values,
            "top_reasons": ["; ".join(r) for r in scores["top_reasons"]],
            "assumed_values": [", ".join(a) for _, a, _ in info],
            "warnings": [" | ".join(w) for _, _, w in info],
        })
        result = result.sort_values("expected_net", ascending=False).reset_index(drop=True)
        result.insert(0, "rank", range(1, len(result) + 1))
    if errors:
        result = pd.concat([result, pd.DataFrame(errors)], ignore_index=True)
    return result


if __name__ == "__main__":
    example = {"recency": 85, "frequency": 3, "monetary": 240, "avg_gap_days": 40, "tenure_days": 200}
    print(json.dumps(predict_customer(example), indent=2))
