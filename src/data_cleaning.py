"""Clean raw transactions: keep a cancellations table, then remove bad and non-product rows."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config
from data_loading import load_transactions


def is_non_product(stock_codes):
    """Return True for rows whose stock code is a fee, postage, adjustment etc., not a product."""
    codes = stock_codes.astype(str)
    mask = codes.isin(config.NON_PRODUCT_CODES)
    for prefix in config.NON_PRODUCT_PREFIXES:
        mask = mask | codes.str.startswith(prefix)
    return mask


def build_cancellations_table(df):
    """Return one row per cancelled invoice line for product rows of known customers.

    We keep this BEFORE cleaning because cleaning throws cancellations away,
    but later we need them to compute each customer's return_rate.
    It keeps invoice_date so that, later, only cancellations before the cutoff are used (no leakage).
    """
    is_cancelled = df["invoice"].str.startswith(config.CANCELLED_INVOICE_PREFIX)
    cancels = df[is_cancelled & df["customer_id"].notna() & ~is_non_product(df["stock_code"])].copy()
    cancels["customer_id"] = cancels["customer_id"].astype(int)
    # Cancelled quantities are negative; store the returned value as a positive amount.
    cancels["cancelled_value"] = (cancels["quantity"] * cancels["price"]).abs()
    return cancels[["invoice", "invoice_date", "customer_id", "stock_code", "quantity", "price", "cancelled_value"]]


def remove_rows(df, mask, reason, log):
    """Drop rows where mask is True, print how many were removed and why, and record it in log."""
    n = int(mask.sum())
    print(f"  - removed {n:>9,} rows: {reason}")
    log.append({"step": reason, "rows_removed": n})
    return df[~mask]


def clean_transactions(df):
    """Return (clean_df, cancellations_df, removal_log) from raw transactions.

    Steps: save cancellations, remove cancelled / bad-debt invoices, missing customers,
    non-product codes, non-positive quantity or price, and exact duplicates.
    """
    print(f"Starting rows: {len(df):,}")
    log = []

    # Step 0: keep cancellations BEFORE they are removed (needed for return_rate later).
    cancellations = build_cancellations_table(df)
    print(f"Cancellations table kept: {len(cancellations):,} product rows, "
          f"{cancellations['invoice'].nunique():,} invoices\n")

    print("Removal steps:")
    df = remove_rows(df, df["invoice"].str.startswith(config.CANCELLED_INVOICE_PREFIX),
                     "cancelled invoices (start with 'C')", log)
    df = remove_rows(df, df["invoice"].str.startswith(config.BAD_DEBT_INVOICE_PREFIX),
                     "bad-debt adjustment invoices (start with 'A')", log)
    df = remove_rows(df, df["customer_id"].isna(), "missing customer_id (guest checkout)", log)

    # Print each non-product code and its row count so it can be checked by hand.
    non_product = is_non_product(df["stock_code"])
    print("    non-product codes found (code: rows):")
    for code, count in df.loc[non_product, "stock_code"].value_counts().items():
        print(f"      {code}: {count:,}")
    df = remove_rows(df, non_product, "non-product stock codes (fees, postage, adjustments...)", log)

    df = remove_rows(df, df["quantity"] <= 0, "quantity <= 0", log)
    df = remove_rows(df, df["price"] <= 0, "price <= 0", log)
    df = remove_rows(df, df.duplicated(), "exact duplicate rows", log)

    df = df.copy()
    df["total_price"] = df["quantity"] * df["price"]
    # IDs were read as text like "13085"; integers are smaller and easier to join on.
    df["customer_id"] = df["customer_id"].astype(int)
    df = df.reset_index(drop=True)

    print(f"\nFinal clean rows: {len(df):,}")
    return df, cancellations, pd.DataFrame(log)


def run_cleaning():
    """Load raw data, clean it, save clean + cancellations Parquet files and the removal log CSV."""
    raw = load_transactions()
    clean, cancellations, log = clean_transactions(raw)
    clean.to_parquet(config.TRANSACTIONS_CLEAN_PARQUET, index=False)
    cancellations.to_parquet(config.CANCELLATIONS_PARQUET, index=False)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    log.to_csv(config.TABLES_DIR / "cleaning_log.csv", index=False)
    print(f"Saved: {config.TRANSACTIONS_CLEAN_PARQUET}")
    print(f"Saved: {config.CANCELLATIONS_PARQUET}")
    return clean, cancellations


if __name__ == "__main__":
    run_cleaning()
