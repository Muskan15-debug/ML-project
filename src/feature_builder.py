"""Build one row of features per customer from transactions BEFORE a given date.

This is the single feature function used by training, testing and the Streamlit app,
so all three compute features in exactly the same way.

Choices for customers with only one order (no gaps between orders exist):
- avg_gap_days = recency. The customer has waited at least this many days without a
  second order, so it is the best lower-bound guess for their gap (and it grows the
  longer they stay away, which is what we want the model to see).
- purchase_regularity = 0. With fewer than two gaps the spread of gaps cannot be measured;
  0 means "no measured variation". The is_single_order flag tells the model these
  two values are filled, not measured.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config

# Final feature column order; the app and the models rely on this list.
FEATURE_COLUMNS = [
    "recency", "frequency", "monetary", "avg_order_value", "avg_gap_days",
    "tenure_days", "unique_products", "return_rate", "spend_trend",
    "purchase_regularity", "is_uk", "is_single_order",
]


def build_order_table(transactions):
    """Collapse line items into one row per order (invoice) with its date and value."""
    return transactions.groupby(["customer_id", "invoice"], as_index=False).agg(
        order_date=("invoice_date", "min"),
        order_value=("total_price", "sum"),
    )


def gap_statistics(orders):
    """Return per customer the mean and standard deviation of days between consecutive orders."""
    orders = orders.sort_values(["customer_id", "order_date"])
    # Gap in days (with fractions) between each order and the previous order of the same customer.
    gaps = orders.groupby("customer_id")["order_date"].diff().dt.total_seconds() / 86400
    gaps = gaps.dropna()
    customer_of_gap = orders.loc[gaps.index, "customer_id"]
    stats = gaps.groupby(customer_of_gap).agg(avg_gap_days="mean", purchase_regularity="std")
    # A customer with exactly 2 orders has one gap: its spread is 0, not unknown.
    stats["purchase_regularity"] = stats["purchase_regularity"].fillna(0.0)
    return stats


def return_rate_table(orders, cancellations):
    """Return per customer: cancelled invoices / (normal orders + cancelled invoices).

    Both inputs must already be limited to dates before as_of_date.
    """
    n_orders = orders.groupby("customer_id")["invoice"].nunique()
    n_cancelled = cancellations.groupby("customer_id")["invoice"].nunique()
    n_cancelled = n_cancelled.reindex(n_orders.index, fill_value=0)
    return (n_cancelled / (n_orders + n_cancelled)).rename("return_rate")


def build_features(transactions, cancellations, as_of_date):
    """Return one row per customer (index = customer_id) with features as of as_of_date.

    LEAKAGE RULE: only rows with invoice_date < as_of_date are used, for both
    purchases and cancellations, so nothing from the future can enter the features.
    """
    as_of_date = pd.Timestamp(as_of_date)
    past = transactions[transactions["invoice_date"] < as_of_date]
    past_cancels = cancellations[cancellations["invoice_date"] < as_of_date]

    orders = build_order_table(past)
    per_customer = orders.groupby("customer_id").agg(
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
        frequency=("invoice", "nunique"),
        monetary=("order_value", "sum"),
    )

    features = pd.DataFrame(index=per_customer.index)
    day = pd.Timedelta(days=1)
    features["recency"] = (as_of_date - per_customer["last_order"]) / day
    features["frequency"] = per_customer["frequency"]
    features["monetary"] = per_customer["monetary"]
    features["avg_order_value"] = per_customer["monetary"] / per_customer["frequency"]
    features["tenure_days"] = (as_of_date - per_customer["first_order"]) / day

    gaps = gap_statistics(orders)
    features = features.join(gaps)
    single = features["frequency"] == 1
    # See module docstring: filled values for single-order customers.
    features.loc[single, "avg_gap_days"] = features.loc[single, "recency"]
    features.loc[single, "purchase_regularity"] = 0.0

    features["unique_products"] = past.groupby("customer_id")["stock_code"].nunique()
    features["return_rate"] = return_rate_table(orders, past_cancels)

    # spend_trend = share of the customer's spend made in the last TREND_WINDOW_MONTHS.
    # Near 1 = recently active / growing; near 0 = spending has dried up.
    trend_start = as_of_date - pd.DateOffset(months=config.TREND_WINDOW_MONTHS)
    recent_spend = orders[orders["order_date"] >= trend_start].groupby("customer_id")["order_value"].sum()
    features["spend_trend"] = recent_spend.reindex(features.index, fill_value=0.0) / features["monetary"]

    # A customer's country is the one they ordered from most often.
    country = past.groupby("customer_id")["country"].agg(lambda s: s.mode().iloc[0])
    features["is_uk"] = (country == config.HOME_COUNTRY).astype(int)
    features["is_single_order"] = single.astype(int)

    return features[FEATURE_COLUMNS]
