"""Plotting helpers for exploratory data analysis. Every function saves its figure to reports/figures/."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
import config


def save_matplotlib(fig, name):
    """Save a Matplotlib/Seaborn figure as PNG in reports/figures/ and return the path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def monthly_revenue(df):
    """Return total revenue per calendar month as a Series indexed by month start."""
    return df.set_index("invoice_date")["total_price"].resample("MS").sum()


def customer_summary(df):
    """Return one row per customer with number of orders and total spend."""
    return df.groupby("customer_id").agg(
        orders=("invoice", "nunique"),
        total_spend=("total_price", "sum"),
    )


def plot_monthly_revenue_matplotlib(df):
    """Line chart of monthly revenue drawn with plain Matplotlib."""
    revenue = monthly_revenue(df)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(revenue.index, revenue.values / 1000, marker="o")
    ax.set_title("Monthly revenue (Matplotlib)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (thousand GBP)")
    ax.grid(alpha=0.3)
    save_matplotlib(fig, "monthly_revenue_matplotlib")
    return fig


def plot_monthly_revenue_seaborn(df):
    """Line chart of monthly revenue drawn with Seaborn."""
    revenue = monthly_revenue(df).reset_index()
    revenue["revenue_k"] = revenue["total_price"] / 1000
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=revenue, x="invoice_date", y="revenue_k", marker="o", ax=ax)
    ax.set_title("Monthly revenue (Seaborn)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (thousand GBP)")
    save_matplotlib(fig, "monthly_revenue_seaborn")
    return fig


def plot_monthly_revenue_plotly(df):
    """Interactive line chart of monthly revenue drawn with Plotly; also saved as PNG."""
    revenue = monthly_revenue(df).reset_index()
    revenue["revenue_k"] = revenue["total_price"] / 1000
    fig = px.line(revenue, x="invoice_date", y="revenue_k", markers=True,
                  title="Monthly revenue (Plotly)",
                  labels={"invoice_date": "Month", "revenue_k": "Revenue (thousand GBP)"})
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(config.FIGURES_DIR / "monthly_revenue_plotly.png", width=1000, height=400)
    return fig


def plot_top_countries(df, n=10):
    """Horizontal bar chart of the n countries with the highest revenue."""
    top = df.groupby("country", observed=True)["total_price"].sum().nlargest(n) / 1000
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top.values, y=top.index.astype(str), ax=ax, color="steelblue")
    ax.set_title(f"Top {n} countries by revenue")
    ax.set_xlabel("Revenue (thousand GBP)")
    ax.set_ylabel("")
    save_matplotlib(fig, "top_countries_revenue")
    return fig, top


def plot_top_products(df, n=10):
    """Horizontal bar chart of the n products with the highest quantity sold."""
    # Group by stock code (descriptions vary slightly for the same product) and show its most common name.
    qty = df.groupby("stock_code")["quantity"].sum().nlargest(n)
    names = df[df["stock_code"].isin(qty.index)].groupby("stock_code")["description"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else "")
    labels = [f"{names[code]} ({code})" for code in qty.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=qty.values / 1000, y=labels, ax=ax, color="seagreen")
    ax.set_title(f"Top {n} products by quantity sold")
    ax.set_xlabel("Quantity (thousand units)")
    ax.set_ylabel("")
    save_matplotlib(fig, "top_products_quantity")
    return fig, qty


def plot_customer_distributions(customers):
    """Histograms of orders per customer and spend per customer, both on a log x-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Log-spaced bins because a few customers order/spend far more than the rest (very skewed).
    order_bins = np.logspace(0, np.log10(customers["orders"].max()), 40)
    axes[0].hist(customers["orders"], bins=order_bins, color="slateblue")
    axes[0].set_xscale("log")
    axes[0].set_title("Orders per customer")
    axes[0].set_xlabel("Number of orders (log scale)")
    axes[0].set_ylabel("Customers")

    spend = customers["total_spend"]
    spend_bins = np.logspace(np.log10(spend.min()), np.log10(spend.max()), 40)
    axes[1].hist(spend, bins=spend_bins, color="darkorange")
    axes[1].set_xscale("log")
    axes[1].set_title("Total spend per customer")
    axes[1].set_xlabel("Total spend in GBP (log scale)")
    axes[1].set_ylabel("Customers")
    save_matplotlib(fig, "customer_orders_spend_hist")
    return fig


def plot_spend_boxplot(customers):
    """Boxplots of customer total spend: raw scale (shows outliers) and log scale (shows shape)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    sns.boxplot(x=customers["total_spend"], ax=axes[0], color="lightcoral")
    axes[0].set_title("Customer total spend (raw)")
    axes[0].set_xlabel("Total spend (GBP)")
    sns.boxplot(x=np.log1p(customers["total_spend"]), ax=axes[1], color="lightcoral")
    axes[1].set_title("Customer total spend (log1p)")
    axes[1].set_xlabel("log(1 + total spend)")
    save_matplotlib(fig, "customer_spend_boxplot")
    return fig
