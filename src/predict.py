"""Predict churn, value, segment and action for typed-in customers (form) or a CSV (batch).

Uses ONLY the saved files written by train.py:
  segmentation.pkl, segment_encoder.pkl, selected_features.json, churn_model.pkl,
  spend_model.pkl (two-part model, via hurdle_model.predict_hurdle), decision_thresholds.json,
  training_medians.json.

Inputs (column / key names), all things a shop owner can look up:
  recency          days since last order
  frequency        number of orders
  monetary         total spent (GBP)
  tenure_days      days since first order (may be left out for a customer with one order)
  unique_products  number of different products bought
  spend_last_3m    spend in the last 3 months (GBP)
  returned_orders  number of returned / cancelled orders
  is_uk            customer in the UK (1 / 0, or yes / no)
  optional, advanced: purchase_regularity (spread of days between orders)

CALCULATED with the same rules as training: avg_order_value, avg_gap_days, spend_trend, return_rate,
is_single_order. Only purchase_regularity can be ASSUMED (TRAINING median), and it is reported as assumed.
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

INPUT_FIELDS = {
    "recency": "Days since last order",
    "frequency": "Number of orders",
    "monetary": "Total spent (GBP)",
    "tenure_days": "Days since first order",
    "unique_products": "Number of different products bought",
    "spend_last_3m": "Spend in the last 3 months (GBP)",
    "returned_orders": "Number of returned / cancelled orders",
    "is_uk": "Customer in the UK (1 = yes, 0 = no)",
}
REQUIRED_FIELDS = list(INPUT_FIELDS)
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


def _to_flag(value):
    """Turn 1/0, True/False or yes/no into 1 or 0."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"yes", "y", "true", "1", "uk"}:
            return 1
        if text in {"no", "n", "false", "0"}:
            return 0
        raise ValueError(f"customer in UK must be yes/no or 1/0, got '{value}'")
    return int(bool(value))


def build_feature_row(inputs, medians):
    """Turn one input dict into the 12 model features.

    Returns (features dict, filled dict {feature: (value, 'derived' or 'assumed', explanation)}, warnings list).
    Raises ValueError for missing or impossible inputs.
    """
    one_order = _given(inputs, "frequency") and float(inputs["frequency"]) == 1
    missing = [INPUT_FIELDS[k] for k in REQUIRED_FIELDS
               if not _given(inputs, k) and not (k == "tenure_days" and one_order)]
    if missing:
        raise ValueError("missing: " + ", ".join(missing))

    recency, frequency, monetary = float(inputs["recency"]), float(inputs["frequency"]), float(inputs["monetary"])
    unique_products, spend_last_3m = float(inputs["unique_products"]), float(inputs["spend_last_3m"])
    returned = float(inputs["returned_orders"])
    if recency < 0:
        raise ValueError("days since last order cannot be negative")
    if frequency < 1 or frequency != int(frequency):
        raise ValueError("number of orders must be a whole number of at least 1")
    if monetary <= 0:
        raise ValueError("total spent must be above 0 GBP")
    if unique_products < 1:
        raise ValueError("number of different products must be at least 1")
    if returned < 0:
        raise ValueError("number of returned orders cannot be negative")
    if spend_last_3m < 0:
        raise ValueError("spend in the last 3 months cannot be negative")
    if spend_last_3m > monetary + 0.01:
        raise ValueError("spend in the last 3 months cannot be larger than total spent")

    features = {"recency": recency, "frequency": frequency, "monetary": monetary,
                "unique_products": unique_products, "is_uk": _to_flag(inputs["is_uk"])}
    filled, warnings = {}, []
    trend_days = config.TREND_WINDOW_MONTHS * config.DAYS_PER_MONTH

    def fill(name, value, kind, why):
        features[name] = value
        filled[name] = (value, kind, why)

    # Days since first order.
    if frequency == 1:
        if _given(inputs, "tenure_days") and abs(float(inputs["tenure_days"]) - recency) > 0.5:
            warnings.append(f"with only one order, days since first order must equal days since last order; "
                            f"using {recency:.0f} instead of {float(inputs['tenure_days']):.0f}")
        fill("tenure_days", recency, "derived", "one order: first order = last order")
    else:
        features["tenure_days"] = float(inputs["tenure_days"])
        # Equal is allowed: some real customers placed several orders on the same day.
        if features["tenure_days"] < recency:
            raise ValueError("days since first order cannot be smaller than days since last order")

    # +1 day margin: three calendar months can be up to 92 days long.
    if recency > trend_days + 1 and spend_last_3m > 0:
        raise ValueError(f"spend in the last 3 months must be 0 when the last order was {recency:.0f} days ago "
                         f"(more than 3 months)")
    if features["tenure_days"] < trend_days - 1 and abs(spend_last_3m - monetary) > 0.01:
        warnings.append("the first order is within the last 3 months, so all spend should be in the last 3 months; "
                        "spend in the last 3 months was typed lower than total spent")

    # Calculated values (same rules as the training feature builder).
    fill("avg_order_value", monetary / frequency, "derived", "total spent / number of orders")
    fill("is_single_order", int(frequency == 1), "derived", "1 if only one order")
    if frequency == 1:
        fill("avg_gap_days", recency, "derived", "one order: gap so far = days since that order (training rule)")
    else:
        fill("avg_gap_days", (features["tenure_days"] - recency) / (frequency - 1), "derived",
             "(days since first order - days since last order) / (orders - 1)")
    fill("spend_trend", spend_last_3m / monetary, "derived", "spend in last 3 months / total spent")
    fill("return_rate", returned / (frequency + returned), "derived", "returned / (orders + returned)")

    # Spread of the gaps between orders: the only value that may be assumed.
    if _given(inputs, "purchase_regularity"):
        features["purchase_regularity"] = float(inputs["purchase_regularity"])
    elif frequency <= 2:
        fill("purchase_regularity", 0.0, "derived", "fewer than 3 orders: no spread can be measured (training rule)")
    else:
        fill("purchase_regularity", medians["purchase_regularity_3plus_orders"], "assumed",
             "cannot be typed; training median for customers with 3+ orders")

    if recency > config.ACTIVE_WINDOW_DAYS:
        warnings.append(WIN_BACK_WARNING)
    if frequency == 1:
        warnings.append("only one order: several values are set by rule, so confidence is lower")
    return {name: features[name] for name in FEATURE_COLUMNS}, filled, warnings


def table_to_inputs(table, include_regularity=False):
    """Turn a customer feature table (e.g. the test table) into form-style inputs, one row per customer."""
    inputs = pd.DataFrame({
        "customer_id": table.index,
        "recency": table["recency"].round(1).values,
        "frequency": table["frequency"].values,
        "monetary": table["monetary"].round(2).values,
        "tenure_days": table["tenure_days"].round(1).values,
        "unique_products": table["unique_products"].values,
        "spend_last_3m": (table["spend_trend"] * table["monetary"]).round(2).values,
        # return_rate = returned / (orders + returned)  ->  returned = rate x orders / (1 - rate)
        "returned_orders": (table["return_rate"] * table["frequency"] / (1 - table["return_rate"])).round().values,
        "is_uk": table["is_uk"].values,
    })
    if include_regularity:
        inputs["purchase_regularity"] = table["purchase_regularity"].values
    return inputs


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
    example = {"recency": 85, "frequency": 3, "monetary": 240, "tenure_days": 200, "unique_products": 6,
               "spend_last_3m": 80, "returned_orders": 0, "is_uk": 1}
    print(json.dumps(predict_customer(example), indent=2))
