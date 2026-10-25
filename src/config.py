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

# ---------- Segmentation ----------
RFM_COLUMNS = ["recency", "frequency", "monetary"]
RFM_LOG_COLUMNS = ["frequency", "monetary"]   # very skewed (skew 8.8 and 21.6), so log1p them
K_RANGE = range(2, 11)                        # k values tried for the elbow / silhouette check
N_SEGMENTS = 4                                # chosen from elbow + silhouette (see notebook 02)
SEGMENTATION_MODEL = MODELS_DIR / "segmentation.pkl"

# ---------- Feature selection ----------
# Features with skew > 2 on TRAIN (frequency 8.8, monetary 21.6, avg_order_value 12.0,
# unique_products 5.9, purchase_regularity 2.1). log1p makes them usable for linear models.
LOG1P_FEATURES = ["frequency", "monetary", "avg_order_value", "unique_products", "purchase_regularity"]
SEGMENT_COLUMN = "segment_name"
N_SELECTED_FEATURES = 8           # each ranking method keeps its top 8
SEGMENT_ENCODER = MODELS_DIR / "segment_encoder.pkl"
SELECTED_FEATURES_JSON = MODELS_DIR / "selected_features.json"

# ---------- Churn models ----------
CV_FOLDS = 5
CV_SCORING = "roc_auc"            # ranking quality; does not depend on a 0.5 threshold
DECISION_THRESHOLD = 0.5          # default threshold for precision/recall/F1 (tuned by cost in Phase 9)
LEAKAGE_ALARM_ROC_AUC = 0.90      # a test ROC-AUC above this is suspicious and must be checked
LIFT_TOP_SHARES = [0.10, 0.20]
CHURN_MODEL = MODELS_DIR / "churn_model.pkl"

# ---------- Spend models ----------
SPEND_CV_SCORING = "neg_root_mean_squared_error"   # RMSE on the log1p target (used for tuning; reported too)
# The FINAL spend model is chosen by out-of-fold CV RMSE in GBP on TRAIN, because the profit rule uses GBP.
HIST_RATE_MIN_TENURE_DAYS = 30    # historical-rate baseline: avoid dividing by a tiny tenure
HIST_RATE_HORIZON_DAYS = 182      # about 6 months, same length as the spend window
LEAKAGE_ALARM_R2 = 0.90           # a test R-squared above this is suspicious and must be checked
SPEND_MODEL = MODELS_DIR / "spend_model.pkl"

TRAIN_TABLE_PARQUET = PROCESSED_DIR / "train_table.parquet"
TEST_TABLE_PARQUET = PROCESSED_DIR / "test_table.parquet"

# ---------- Reproducibility ----------
RANDOM_STATE = 42

# Parallel jobs for GridSearchCV. 1 = sequential: parallel workers were killed when the laptop was low on free RAM.
N_JOBS = 1

# ---------- Business assumptions (NOT from data; change if you have real values) ----------
# The offer is a discount. Its real cost is the discount given away, including to customers who
# would have stayed anyway, plus a small cost per contact.
GROSS_MARGIN = 0.30         # ASSUMPTION: profit share of each GBP of sales
DISCOUNT_RATE = 0.10        # ASSUMPTION: discount given with the retention offer (10% off)
CONTACT_COST = 1.0          # ASSUMPTION: GBP cost of contacting one customer
OFFER_RESPONSE_RATE = 0.20  # ASSUMPTION: share of would-be churners the offer actually retains
STAYER_REDEEM_RATE = 1.0    # ASSUMPTION: share of non-churners who use the discount anyway (worst case)
SENSITIVITY_RESPONSE_RATES = [0.05, 0.10, 0.20, 0.30]
SENSITIVITY_DISCOUNT_RATES = [0.05, 0.10, 0.20]
TOP_N_SHARE = 0.20          # strategy (b): contact the 20% highest churn-risk customers
PRIORITY_LIST_SIZE = 50

# ---------- Explanations ----------
SHAP_SAMPLE_SIZE = 300      # test customers used for SHAP plots (keeps memory low)
CART_RULES_DEPTH = 3        # depth of the readable CART tree
DECISION_THRESHOLDS_JSON = MODELS_DIR / "decision_thresholds.json"

# Plain-English names for features, used in plots, explanations and the app.
FEATURE_LABELS = {
    "recency": "Days since last order",
    "frequency": "Number of orders",
    "monetary": "Total spent (GBP)",
    "avg_order_value": "Average order value (GBP)",
    "avg_gap_days": "Average days between orders",
    "tenure_days": "Days since first order",
    "unique_products": "Different products bought",
    "return_rate": "Share of orders returned",
    "spend_trend": "Share of spend in last 3 months",
    "purchase_regularity": "Variation in days between orders",
    "is_uk": "UK customer",
    "is_single_order": "Only one order",
    "seg_Champions": "In Champions segment",
    "seg_Loyal": "In Loyal segment",
    "seg_One-off buyers": "In One-off buyers segment",
    "seg_At-Risk": "In At-Risk segment",
}
