"""Build the two targets (churn and future spend) from transactions AFTER a cutoff date."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config


def label_window_ends(cutoff_date):
    """Return (churn_window_end, spend_window_end) for a cutoff. Both ends are exclusive."""
    cutoff_date = pd.Timestamp(cutoff_date)
    churn_end = cutoff_date + pd.Timedelta(days=config.CHURN_WINDOW_DAYS)
    spend_end = cutoff_date + pd.DateOffset(months=config.SPEND_WINDOW_MONTHS)
    return churn_end, spend_end


def build_labels(transactions, cutoff_date):
    """Return per customer: churn (1 = no purchase in the churn window) and future_spend.

    Only customers who bought something BEFORE the cutoff are included, because
    those are the only customers we can build features for.
    Labels use only rows ON or AFTER the cutoff, so they never overlap the feature data.
    """
    cutoff_date = pd.Timestamp(cutoff_date)
    churn_end, spend_end = label_window_ends(cutoff_date)
    dates = transactions["invoice_date"]

    known_customers = transactions.loc[dates < cutoff_date, "customer_id"].unique()
    labels = pd.DataFrame(index=pd.Index(sorted(known_customers), name="customer_id"))

    in_churn_window = transactions[(dates >= cutoff_date) & (dates < churn_end)]
    bought_again = labels.index.isin(in_churn_window["customer_id"].unique())
    # Label creation: churn = 1 means the customer did NOT come back within the window.
    labels["churn"] = (~bought_again).astype(int)

    in_spend_window = transactions[(dates >= cutoff_date) & (dates < spend_end)]
    spend = in_spend_window.groupby("customer_id")["total_price"].sum()
    labels["future_spend"] = spend.reindex(labels.index, fill_value=0.0)
    return labels
