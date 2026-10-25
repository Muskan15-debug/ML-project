"""Sanity checks on the final churn model's behaviour (does it react sensibly to recency?)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config
from eda import save_matplotlib
from predict import predict_customer

RECENCY_VALUES = [5, 30, 60, 90, 120, 180]

# Four typical customers. The first order date is fixed (tenure stays the same) while recency changes,
# except the one-order customer, whose first order IS the last order.
PROFILES = {
    "1 order": {"frequency": 1, "monetary": 300, "unique_products": 15, "tenure_days": None, "recent_share": 1.0},
    "3 orders": {"frequency": 3, "monetary": 900, "unique_products": 40, "tenure_days": 250, "recent_share": 0.3},
    "8 orders": {"frequency": 8, "monetary": 3000, "unique_products": 80, "tenure_days": 300, "recent_share": 0.3},
    "20 orders": {"frequency": 20, "monetary": 10000, "unique_products": 150, "tenure_days": 330, "recent_share": 0.3},
}


def profile_inputs(profile, recency):
    """Form inputs for one profile at one recency. Spend in the last 3 months is 0 once recency > 3 months."""
    trend_days = config.TREND_WINDOW_MONTHS * config.DAYS_PER_MONTH
    recent = profile["monetary"] * profile["recent_share"] if recency <= trend_days else 0.0
    return {"recency": recency, "frequency": profile["frequency"], "monetary": profile["monetary"],
            "tenure_days": profile["tenure_days"] if profile["tenure_days"] else recency,
            "unique_products": profile["unique_products"], "spend_last_3m": recent,
            "returned_orders": 0, "is_uk": 1}


def recency_table(recency_values=None, predict_fn=predict_customer):
    """Churn probability for each profile x recency value (rows = recency, columns = profile)."""
    recency_values = recency_values or RECENCY_VALUES
    table = pd.DataFrame(index=pd.Index(recency_values, name="recency_days"))
    for name, profile in PROFILES.items():
        table[name] = [predict_fn(profile_inputs(profile, r))["churn_probability"] for r in recency_values]
    return table


def decreasing_steps(table, tolerance=0.0):
    """List every place where churn probability goes DOWN when recency goes up."""
    problems = []
    for name in table.columns:
        diffs = table[name].diff().dropna()
        for recency, change in diffs.items():
            if change < -tolerance:
                previous = table.index[table.index.get_loc(recency) - 1]
                problems.append({"profile": name, "from_recency": previous, "to_recency": recency,
                                 "change": round(change, 4)})
    return pd.DataFrame(problems, columns=["profile", "from_recency", "to_recency", "change"])


def plot_recency(table, file_name, title):
    """Line plot of churn probability against recency for each profile."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in table.columns:
        ax.plot(table.index, table[name], marker="o", label=name)
    ax.set_xlabel("Days since last order")
    ax.set_ylabel("Predicted churn probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    save_matplotlib(fig, file_name)
    return fig
