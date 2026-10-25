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

## Train all models

```bash
python src/train.py
```

Runs everything from the raw file to the saved models in `models/` (about 1.5 minutes):
cleaning (if needed), train/test tables, K-Means segments, feature selection, churn model,
two-part spend model, decision thresholds and the training medians used by the app.

## Predict

```python
from predict import predict_customer, predict_batch   # run with src/ on the path

predict_customer({"recency": 85, "frequency": 3, "monetary": 240, "tenure_days": 200})
predict_batch(pandas_dataframe_with_the_same_columns)
```

Required inputs: `recency`, `frequency`, `monetary`. Optional: `tenure_days`, `avg_gap_days`,
`unique_products`, `return_rate`, `is_uk`. Values that are not given are derived with the training
rules or filled with training medians, and the result lists which ones were assumed.

## Tests

```bash
python -m pytest tests
```
