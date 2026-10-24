"""All project settings in one place. Other files import from here; no magic numbers elsewhere."""

from pathlib import Path

import pandas as pd

# ---------- Paths ----------
# PROJECT_ROOT is the folder that contains src/, so paths work no matter where we run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# The raw file can be the original Excel (.xlsx, two sheets) or a CSV export.
RAW_FILE_CSV = RAW_DIR / "online_retail_II.csv"
RAW_FILE_XLSX = RAW_DIR / "online_retail_II.xlsx"
EXCEL_SHEETS = ["Year 2009-2010", "Year 2010-2011"]

TRANSACTIONS_RAW_PARQUET = PROCESSED_DIR / "transactions_raw.parquet"
TRANSACTIONS_CLEAN_PARQUET = PROCESSED_DIR / "transactions_clean.parquet"
CANCELLATIONS_PARQUET = PROCESSED_DIR / "cancellations.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# ---------- Column names ----------
# Original column name -> our clean snake_case name.
COLUMN_RENAME = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# ---------- Cleaning ----------
# Stock codes that are fees, postage, discounts, samples, tests etc. - not real products.
# Chosen by checking every code that does not start with 5 digits (see Phase 1 output).
# DCGS* and SP1002 are real products and are deliberately NOT in this list.
NON_PRODUCT_CODES = [
    "POST", "DOT", "M", "m", "C2", "C3", "D", "S", "B",
    "BANK CHARGES", "ADJUST", "ADJUST2", "AMAZONFEE", "PADS",
    "CRUK", "TEST001", "TEST002", "GIFT",
]
# Gift vouchers look like gift_0001_10, gift_0001_20, ... - matched by prefix.
NON_PRODUCT_PREFIXES = ["gift_"]

CANCELLED_INVOICE_PREFIX = "C"     # invoices starting with C are cancellations/returns
BAD_DEBT_INVOICE_PREFIX = "A"      # invoices starting with A are bad-debt adjustments

# ---------- Time split ----------
# Two snapshots give an out-of-time test: train on the earlier one, test on the later one.
# Features use only data BEFORE a cutoff; labels use only data AFTER it (no leakage).
TRAIN_CUTOFF = pd.Timestamp("2010-12-01")
TEST_CUTOFF = pd.Timestamp("2011-06-01")

CHURN_WINDOW_DAYS = 90      # churn = no purchase in this many days after the cutoff
SPEND_WINDOW_MONTHS = 6     # future spend = total spend in this many months after the cutoff

# Customer base = ACTIVE customers only: at least one purchase in this many days before the cutoff.
# Customers gone for a year or more are trivially "churned" and would inflate the churn rate.
# Features still use the full history; only the set of customers is filtered.
ACTIVE_WINDOW_DAYS = 180

# Candidate train cutoffs compared in the Phase 2 seasonality check.
CANDIDATE_TRAIN_CUTOFFS = [
    pd.Timestamp("2010-06-01"),
    pd.Timestamp("2010-09-01"),
    pd.Timestamp("2010-12-01"),
]

# ---------- Features ----------
TREND_WINDOW_MONTHS = 3     # spend_trend compares the last 3 months with the earlier history
HOME_COUNTRY = "United Kingdom"

TRAIN_TABLE_PARQUET = PROCESSED_DIR / "train_table.parquet"
TEST_TABLE_PARQUET = PROCESSED_DIR / "test_table.parquet"

# ---------- Reproducibility ----------
RANDOM_STATE = 42

# ---------- Business assumptions (NOT from data; change if you have real values) ----------
OFFER_COST = 5.0            # ASSUMPTION: cost in GBP of sending one retention offer
OFFER_RESPONSE_RATE = 0.20  # ASSUMPTION: share of contacted churners the offer actually retains
