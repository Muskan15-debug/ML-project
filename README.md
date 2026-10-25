# Customer Lifetime Value & Churn Predictor

ML Lab project (20CP401P). Predicts which customers will churn (no purchase in the next 90 days),
how much they will spend in the next 6 months, and whether a retention discount is worth sending.
Includes a Streamlit app for single customers, CSV batches and a campaign simulator.

- Full specification: [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)
- Results with all numbers: [reports/RESULTS_SUMMARY.md](reports/RESULTS_SUMMARY.md)
- Report draft: [reports/REPORT_DRAFT.md](reports/REPORT_DRAFT.md)
- Viva questions: [reports/VIVA_QA.md](reports/VIVA_QA.md)
- Known issues and the main business finding: [reports/RESULTS_NOTES.md](reports/RESULTS_NOTES.md)

## 1. Install

Requires Python 3.10+ (tested on Python 3.13, Windows, 8-12 GB RAM, no GPU).

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 2. Get the data

Download **Online Retail II** (UCI) and place it in `data/raw/` as either
`online_retail_II.xlsx` (both sheets are read) or `online_retail_II.csv`.

## 3. Train all models

```bash
python src/train.py
```

Runs everything from the raw file to the saved models in `models/` (about 2 minutes):
cleaning, train/test tables (cutoffs 2010-12-01 and 2011-06-01, active customers only), K-Means segments,
feature selection, the churn model (XGBoost with monotonic constraints), the two-part spend model,
decision thresholds and the training medians used by the app. It ends with a sanity check on the test snapshot.

All settings (paths, cutoffs, windows, random seed, business assumptions) are in `src/config.py`.
The business values (margin, discount, response rate, redeem rate, contact cost) are ASSUMPTIONS.

## 4. Run the app

```bash
streamlit run app.py
```

Then open http://localhost:8501. Tabs:
1. **Check a Customer** - form (two example presets) with verdict, churn probability, risk level, predicted spend,
   segment, SHAP reasons and recommended action, plus What-If sliders that update live.
2. **Batch Prediction** - upload a CSV (a sample CSV can be downloaded in the tab), get a ranked "contact first" table
   and download the results.
3. **Dashboard** - main business finding, campaign simulator, segments, revenue at risk, model comparison and SHAP.

CSV / form fields: `recency`, `frequency`, `monetary`, `tenure_days`, `unique_products`, `spend_last_3m`,
`returned_orders`, `is_uk` (optional `customer_id`).

## 5. Use the predictions in Python

```python
import sys; sys.path.append("src")
from predict import predict_customer, predict_batch

predict_customer({"recency": 60, "frequency": 8, "monetary": 3000, "tenure_days": 300,
                  "unique_products": 80, "spend_last_3m": 900, "returned_orders": 0, "is_uk": 1})
```

## 6. Tests

```bash
python -m pytest tests
```

35 tests: data-leakage checks (features never use data on or after the cutoff, label windows do not overlap,
only active customers) and prediction checks (types, ranges, invalid inputs, and the app reproducing the training
pipeline's probabilities).

## 7. Notebooks

Run in order; each only calls functions from `src/`:

| Notebook | Content |
|---|---|
| 01_cleaning_and_eda | Cleaning steps and EDA charts |
| 02_features_and_segmentation | Cutoff choice, train/test tables, K-Means segments |
| 03_feature_selection | Filter, wrapper and embedded selection |
| 04_imbalance_and_models | Churn models x imbalance methods, final model, recency sanity check |
| 05_tuning_and_evaluation | Spend models, smearing, two-part model |
| 06_shap_and_business_rules | SHAP, CART rules, action matrix, campaign backtest and sensitivity |

## 8. Project structure

```
data/raw/            original dataset (not in git)
data/processed/      cleaned transactions and train/test tables (not in git)
models/              saved models and settings written by src/train.py
notebooks/           01-06 analysis notebooks
reports/figures/     all figures (PNG)
reports/tables/      all result tables (CSV)
src/                 all reusable code (config, cleaning, features, labels, models, predict, explain, decisions)
tests/               pytest tests
app.py               Streamlit app
```
