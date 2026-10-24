"""Load the raw Online Retail II data, standardise it, save it as Parquet and print a data check."""

import sys
from pathlib import Path

import pandas as pd

# Allow running as "python src/data_loading.py" as well as importing from notebooks.
sys.path.append(str(Path(__file__).resolve().parent))
import config


def read_raw_file():
    """Read the raw data from Excel (both sheets) or CSV, whichever exists."""
    # Reading customer_id and invoice as text keeps values like "C489449" and avoids 13085.0 floats.
    dtypes = {"Invoice": str, "StockCode": str, "Description": str, "Customer ID": str, "Country": str}

    if config.RAW_FILE_XLSX.exists():
        sheets = pd.read_excel(config.RAW_FILE_XLSX, sheet_name=config.EXCEL_SHEETS, dtype=dtypes)
        df = pd.concat(sheets.values(), ignore_index=True)
    elif config.RAW_FILE_CSV.exists():
        df = pd.read_csv(config.RAW_FILE_CSV, dtype=dtypes)
    else:
        raise FileNotFoundError(f"No raw file found in {config.RAW_DIR}")
    return df


def standardise(df):
    """Rename columns to snake_case and give each column the right data type."""
    df = df.rename(columns=config.COLUMN_RENAME)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    # IDs were stored as floats in the source ("13085.0"); strip the ".0" so IDs are clean text.
    df["customer_id"] = df["customer_id"].str.replace(r"\.0$", "", regex=True)
    # Category type saves a lot of memory for columns with few distinct values.
    df["country"] = df["country"].astype("category")
    return df


def load_transactions(force_reload=False):
    """Return the standardised transactions, reading the cached Parquet file if it exists."""
    if config.TRANSACTIONS_RAW_PARQUET.exists() and not force_reload:
        return pd.read_parquet(config.TRANSACTIONS_RAW_PARQUET)

    df = standardise(read_raw_file())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.TRANSACTIONS_RAW_PARQUET, index=False)
    return df


def data_check(df):
    """Print basic facts about the data: size, dates, customers, missing values, bad rows."""
    print(f"Rows:                     {len(df):,}")
    print(f"Date range:               {df['invoice_date'].min()}  to  {df['invoice_date'].max()}")
    print(f"Unique customers:         {df['customer_id'].nunique():,}")
    print(f"Unique invoices:          {df['invoice'].nunique():,}")

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    cancelled = df["invoice"].str.startswith("C")
    print(f"\nCancelled rows (invoice starts with 'C'): {cancelled.sum():,}")
    print(f"Cancelled invoices (distinct):            {df.loc[cancelled, 'invoice'].nunique():,}")
    print(f"Rows with quantity <= 0:                  {(df['quantity'] <= 0).sum():,}")
    print(f"Rows with price <= 0:                     {(df['price'] <= 0).sum():,}")
    print(f"Fully duplicated rows:                    {df.duplicated().sum():,}")


if __name__ == "__main__":
    transactions = load_transactions(force_reload=True)
    print(f"Saved: {config.TRANSACTIONS_RAW_PARQUET}\n")
    data_check(transactions)
