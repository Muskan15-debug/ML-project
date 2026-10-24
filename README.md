# Customer Lifetime Value & Churn Predictor

ML Lab project (20CP401P). Predicts which customers will churn (no purchase in the next 90 days),
how much they will spend in the next 6 months, and what retention action to take.
Full specification: [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md).

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Data

Download **Online Retail II** (UCI) and place it in `data/raw/` as either
`online_retail_II.xlsx` (both sheets are read) or `online_retail_II.csv`.

Then convert it once to a fast Parquet file and print a data check:

```bash
python src/data_loading.py
```

This creates `data/processed/transactions_raw.parquet`.

## Settings

All paths, cutoff dates, window lengths, the random seed and business assumptions
(offer cost, response rate) are in `src/config.py`.
