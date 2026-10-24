# Customer Lifetime Value and Churn Predictor

A machine learning system that predicts **whether a customer will leave (churn)**, **how much they are likely to spend in the future (lifetime value)**, and **what action the business should take**. It includes an interactive web app where you can type in any customer's statistics and instantly see whether that customer is likely to stay or leave, and why.

> **Course:** Machine Learning Lab (20CP401P), B.Tech Computer Engineering, Semester VII
> **Core model:** XGBoost (classifier for churn, regressor for spend), benchmarked against Logistic Regression, CART Decision Tree and Random Forest

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Key Features](#3-key-features)
4. [Dataset](#4-dataset)
5. [System Architecture](#5-system-architecture)
6. [Methodology (Step by Step)](#6-methodology-step-by-step)
7. [Models Trained](#7-models-trained)
8. [Mapping to Lab Experiments](#8-mapping-to-lab-experiments)
9. [Predicting for a Specific Customer](#9-predicting-for-a-specific-customer)
10. [Handling New Customers and New Data](#10-handling-new-customers-and-new-data)
11. [Evaluation Plan](#11-evaluation-plan)
12. [Project Structure](#12-project-structure)
13. [Installation and Usage](#13-installation-and-usage)
14. [Streamlit App Walkthrough](#14-streamlit-app-walkthrough)
15. [Skills Learned Beyond the Labs](#15-skills-learned-beyond-the-labs)
16. [Timeline](#16-timeline)
17. [Limitations](#17-limitations)
18. [Future Enhancements](#18-future-enhancements)
19. [Expected Deliverables](#19-expected-deliverables)
20. [References](#20-references)

---

## 1. Problem Statement

Acquiring a new customer typically costs 5 to 7 times more than keeping an existing one. But a business cannot offer discounts to everyone: that wastes money on customers who would have stayed anyway, and ignores the valuable customers who are quietly leaving.

Companies therefore need to answer two questions:

1. **Who is about to leave?** (churn risk)
2. **Who is worth keeping?** (expected future value)

This project builds a predictive system that answers both and converts the answer into a decision, for example:

> *"Contact these 200 customers first. They are very likely to leave AND are expected to spend the most."*

---

## 2. Objectives

1. Clean and explore a large real-world transaction dataset.
2. Construct churn labels and spend targets from raw transactions without data leakage.
3. Engineer customer-level features (RFM and behavioural features).
4. Segment customers using K-Means clustering.
5. Select the most useful features using several feature-selection techniques.
6. Handle class imbalance with oversampling (SMOTE) and class weights.
7. Train, tune and compare multiple classifiers and regressors.
8. Explain predictions with SHAP so every decision has a reason.
9. Convert predictions into profit-based business actions.
10. Build a web app that predicts for **any customer from typed-in statistics**.

---

## 3. Key Features

| Feature | Description |
|---|---|
| Churn prediction | Probability that a customer makes no purchase in the next 90 days |
| Lifetime value prediction | Expected spend of the customer over the next 6 months |
| Customer segmentation | K-Means groups: Champions, Loyal, At-Risk, Lost |
| Manual customer check | Enter statistics in a form and get stay/leave verdict instantly |
| "What if" sliders | Change one value (e.g., recency) and watch the risk change live |
| Explainability | SHAP chart showing which factors raised or lowered the risk |
| Action recommendation | Profit rule decides whether a retention offer is worthwhile |
| Batch prediction | Upload a CSV of customers and download predictions |
| New customer support | Works for existing, newly added and brand-new customers |
| Priority list | Ranked table of customers to contact first |

---

## 4. Dataset

**Online Retail II** (UCI Machine Learning Repository / Kaggle)

- About 1 million transactions from a UK-based online gift retailer
- Period: December 2009 to December 2011
- Columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`

The dataset has **no ready-made churn label**. Creating the label from raw transactions (Step 2 of the methodology) is one of the most valuable parts of the project.

**Alternative datasets:** Telco Customer Churn (Kaggle), Bank Customer Churn (Kaggle), or any e-commerce transaction log.

---

## 5. System Architecture

```
 Raw transactions (CSV)
          |
          v
 +-------------------+
 |  Data Cleaning    |  remove cancellations, nulls, negatives
 +---------+---------+
           |
           v
 +-------------------+      Observation window  -> features
 |  Time Split       |      Prediction window   -> labels
 +---------+---------+
           |
           v
 +-------------------+
 | Feature Builder   |  RFM + behavioural features per customer
 +---------+---------+
           |
   +-------+---------+
   |                 |
   v                 v
+--------+     +-----------------+
|K-Means |     | Feature         |
|Segments|     | Selection       |
+---+----+     +--------+--------+
    |                   |
    +---------+---------+
              v
      +---------------+
      | Imbalance     |  SMOTE / class weights
      | Handling      |
      +-------+-------+
              v
 +----------------------------+
 | Model Training & Tuning    |
 | LogReg | CART | RF | XGBoost|
 +-------------+--------------+
               v
 +----------------------------+
 | Evaluation + SHAP          |
 +-------------+--------------+
               v
 +----------------------------+
 | Saved Pipeline (joblib)    |
 +-------------+--------------+
               v
 +----------------------------+
 | Streamlit App              |
 | form | CSV upload | what-if|
 +----------------------------+
```

---

## 6. Methodology (Step by Step)

### Step 1: Data Cleaning and EDA
- Remove cancelled invoices (invoice numbers starting with `C`), rows with missing `Customer ID`, and rows with zero or negative quantity or price.
- Create `TotalPrice = Quantity x Price`.
- Visualize monthly revenue, top countries, top products, and the distribution of orders per customer.
- Check outliers and skewness; apply a log transform to skewed monetary values.

### Step 2: Define Targets Without Data Leakage
Choose a **cutoff date** (e.g., 1 June 2011) and split time into two windows:

```
|-------- Observation window --------|---- Prediction window ----|
Dec 2009                        Cutoff date                  +90 days
(features are built here)                           (labels are built here)
```

- **Churn label:** `1` if the customer makes **no purchase** in the 90 days after the cutoff, else `0`.
- **Future spend label:** total spend of the customer in the 6 months after the cutoff.
- Only customers who purchased **before** the cutoff are included.

> **Why this matters:** If any information from after the cutoff leaks into the features, the model looks excellent in testing but fails in real use. Using a strict time split prevents this.

### Step 3: Feature Engineering (Customer-Level Table)
Convert transactions into **one row per customer**:

| Feature | Meaning |
|---|---|
| `recency` | Days since last purchase (before cutoff) |
| `frequency` | Number of distinct orders |
| `monetary` | Total amount spent |
| `avg_order_value` | Monetary / frequency |
| `avg_gap_days` | Average days between consecutive orders |
| `tenure_days` | Days since the customer's first purchase |
| `unique_products` | Number of different products bought |
| `return_rate` | Share of orders that were cancelled/returned |
| `spend_trend` | Spend in last 3 months vs. earlier months |
| `purchase_regularity` | Standard deviation of the gaps between orders |
| `country_group` | UK vs. international |

### Step 4: Customer Segmentation with K-Means
- Scale RFM features with `StandardScaler`.
- Choose `k` using the **elbow method** and **silhouette score**.
- Name the clusters from their averages (e.g., high frequency + low recency = *Champions*; high recency + low frequency = *Lost*).
- Add the **segment** as a feature for the supervised models and report the churn rate in each segment.
- Visualize clusters with PCA (2D scatter plot).

### Step 5: Feature Selection
Compare several approaches and keep the best 12-15 features:

- **Filter methods:** correlation matrix, mutual information, ANOVA F-test
- **Wrapper method:** Recursive Feature Elimination (RFE)
- **Embedded method:** Random Forest / XGBoost importance, Lasso

Show that the reduced model performs almost as well as the full one while being simpler.

### Step 6: Handle Class Imbalance
- Check the churn ratio (churners are often the minority or majority depending on the cutoff).
- Compare **SMOTE** oversampling and **class weights** (`scale_pos_weight` in XGBoost).
- Apply SMOTE **only to the training set**, never to validation/test data.

### Step 7: Train, Tune and Compare Models
- **Classifiers:** Logistic Regression, CART Decision Tree, Random Forest, XGBoost
- **Regressors (spend):** Linear Regression, Random Forest Regressor, XGBoost Regressor
- Tune XGBoost with `GridSearchCV` or **Optuna** (depth, learning rate, number of trees, subsample, regularization).
- Use **time-aware cross-validation**, not random shuffling.

### Step 8: Explain the Model
- Global view: SHAP summary plot of the most important features.
- Local view: SHAP waterfall plot explaining one customer's prediction.
- Show the CART tree's rules as a simple, human-readable baseline.

### Step 9: Turn Predictions into Business Decisions
Combine churn probability and predicted value:

| | **Low churn risk** | **High churn risk** |
|---|---|---|
| **High value** | Keep happy (loyalty perks) | **Priority: retention offer** |
| **Low value** | No action | Low-cost email only |

**Profit rule:** send an offer only if

```
churn_probability x predicted_value  >  cost_of_offer
```

Tune the decision threshold using cost: missing a valuable churner is much more expensive than sending a wasted email.

### Step 10: Save the Pipeline
Save the feature builder, scaler, K-Means model and XGBoost models with `joblib`, so the app applies **exactly the same steps** to new data as were used in training.

---

## 7. Models Trained

| Task | Models compared | Final model |
|---|---|---|
| Churn classification | Logistic Regression, CART, Random Forest, **XGBoost** | XGBoost Classifier |
| Future spend regression | Linear Regression, Random Forest, **XGBoost** | XGBoost Regressor |
| Segmentation | K-Means (k chosen by elbow and silhouette) | K-Means |
| Cold-start (optional) | Logistic Regression / small XGBoost on first-order data | Small XGBoost |

---

## 8. Mapping to Lab Experiments

| Lab | Topic | Used in this project for |
|---|---|---|
| Lab 1 | Data visualization tools | Charts of revenue, countries, churn by segment (Matplotlib, Seaborn, Plotly) |
| Lab 2 | Household power consumption analysis | Same EDA workflow: cleaning, time-based aggregation, trends |
| Lab 3 | Simple and multi-linear regression | Baseline model for predicting future spend |
| Lab 4 | Locally weighted regression | Optional comparison model for the spend prediction |
| Lab 5 | Feature selection algorithms | Filter, wrapper and embedded selection on customer features |
| Lab 6 | Evaluating ML approaches for classification | Comparing Logistic Regression, CART, Random Forest, XGBoost |
| Lab 7 | Oversampling imbalanced dataset | SMOTE vs. class weights on the churn label |
| Lab 8 | Feature selection for classification and regression | Selecting features separately for the churn and spend models |
| Lab 9 | CART decision tree | Explainable baseline tree with readable rules |
| Lab 10 | K-Means clustering | RFM customer segmentation |

---

## 9. Predicting for a Specific Customer

This is the **"Check a Customer"** feature: type in a customer's statistics and see whether they will stay or leave.

### Inputs (form fields)
| Field | Example |
|---|---|
| Days since last purchase (recency) | 85 |
| Number of orders | 3 |
| Total amount spent | 240 |
| Average order value | 80 |
| Average days between orders | 40 |
| Days since first purchase (tenure) | 200 |
| Number of different products | 6 |
| Return rate | 0.0 |

Features that cannot be typed directly (such as `spend_trend`) are derived from the others or filled with training-set medians, and the app shows which values were assumed.

### Outputs
- **Verdict:** "Likely to stay" or "Likely to leave"
- **Churn probability:** e.g., 82%
- **Risk level:** Low / Medium / High
- **Predicted 6-month spend:** e.g., 60
- **Segment:** e.g., At-Risk
- **SHAP explanation:** which values pushed the risk up or down
- **Recommended action:** from the profit rule

### Example

```
Input:   Recency 85 | Orders 3 | Spend 240 | Avg gap 40 | Tenure 200

Output:  Churn probability: 82%      (High risk)
         Segment: At-Risk
         Predicted 6-month value: 60
         Verdict: Likely to leave, low value
         Action: Send low-cost email only

Why:     + 85 days since last order   (raises risk)
         + only 3 orders              (raises risk)
         - 200-day tenure             (lowers risk slightly)
```

### "What If" sliders
Move a slider (e.g., recency from 85 down to 20) and the probability, segment and action update immediately. This makes the model's behaviour easy to explain during a demo or viva.

---

## 10. Handling New Customers and New Data

| Case | How it is handled |
|---|---|
| **New transactions for an existing customer** | Append to the transaction table; the feature builder recomputes recency, frequency, spend and trend, then predicts again |
| **New customer with 1-2 orders** | Same feature builder; undefined features (e.g., average gap) use defaults, an `is_new_customer` flag is added, and a low-confidence warning is shown |
| **Brand-new customer with no orders (cold start)** | A small second model uses first-order information only (country, first basket size, first product category, day of week) to predict whether the customer will become a repeat buyer |
| **Many new customers at once** | Upload a CSV in the app for batch predictions |
| **Model becoming outdated (drift)** | A retraining script refits on all data and replaces the old model only if it performs better on the latest period |

---

## 11. Evaluation Plan

### Churn classification
| Metric | Why |
|---|---|
| Precision, Recall, F1 | Balance between false alarms and missed churners |
| ROC-AUC | Overall ranking quality |
| PR-AUC | More informative under class imbalance |
| Confusion matrix | See the types of errors |
| Lift and gain charts | "Top 10% of risky customers contain X% of all churners" |

### Spend regression
| Metric | Why |
|---|---|
| MAE, RMSE | Average prediction error in currency units |
| R-squared | Share of variance explained |

### Segmentation
| Metric | Why |
|---|---|
| Silhouette score | Cluster separation |
| Elbow plot | Choice of k |
| Churn rate per segment | Business usefulness |

### Business impact
- Compare **campaign cost and retained revenue** for: (a) mailing everyone, (b) mailing the top risky customers, (c) mailing by profit rule.
- Report the savings of the model-driven strategy.

### Baselines to beat
- Logistic Regression and CART for churn
- Linear Regression for spend

---

## 12. Project Structure

```
churn-clv-predictor/
├── data/
│   ├── raw/                    # original Online Retail II files
│   └── processed/              # customer-level feature table
├── models/
│   ├── feature_builder.pkl
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   ├── churn_xgb.pkl
│   ├── spend_xgb.pkl
│   └── coldstart_model.pkl
├── notebooks/
│   ├── 01_cleaning_and_eda.ipynb
│   ├── 02_features_and_segmentation.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_imbalance_and_models.ipynb
│   ├── 05_tuning_and_evaluation.ipynb
│   └── 06_shap_and_business_rules.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_builder.py      # reusable feature function
│   ├── segmentation.py
│   ├── train.py
│   ├── predict.py              # single and batch prediction
│   ├── explain.py              # SHAP helpers
│   ├── decision_rules.py       # profit rule and actions
│   └── retrain.py              # periodic retraining script
├── app.py                      # Streamlit app
├── requirements.txt
└── README.md
```

---

## 13. Installation and Usage

```bash
# 1. Clone the repository and create an environment
git clone <your-repo-url>
cd churn-clv-predictor
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset in data/raw/

# 4. Build features and train the models
python src/train.py

# 5. Launch the app
streamlit run app.py
```

Suggested `requirements.txt`:

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
optuna
matplotlib
seaborn
plotly
streamlit
joblib
openpyxl
```

---

## 14. Streamlit App Walkthrough

The app has four tabs:

1. **Check a Customer**
   Fill in the form (or use sliders) and get the verdict, probability, segment, predicted value, SHAP explanation and recommended action.

2. **What-If Simulator**
   Change one statistic and watch the churn probability move in real time.

3. **Batch Prediction**
   Upload a CSV of customers, receive a ranked "contact first" table, and download the results.

4. **Dashboard**
   Segment map, churn rate per segment, revenue at risk, model comparison table, and feature importance.

---

## 15. Skills Learned Beyond the Labs

| Area | What you will learn |
|---|---|
| Problem framing | Turning raw transactions into a prediction problem with a defined target |
| Data leakage | Time-based splits and leak-free feature engineering |
| Feature engineering | RFM and behavioural features from event data |
| Model tuning | Optuna / GridSearchCV with time-aware cross-validation |
| Gradient boosting | How XGBoost works and why it often wins on tabular data |
| Explainable AI | SHAP for global and per-customer explanations |
| Business metrics | Lift, gain, cost-sensitive thresholds and profit-based decisions |
| Productionizing | Saved pipelines, new-data handling, cold-start and retraining |
| App development | Streamlit forms, sliders, file upload and caching |

---

## 16. Timeline

| Week | Tasks |
|---|---|
| **Week 1** | Download data, cleaning, EDA, RFM features, K-Means segmentation |
| **Week 2** | Churn and spend labels, reusable feature function, feature selection, baselines, imbalance handling |
| **Week 3** | XGBoost tuning, model comparison, SHAP, profit rule, saved pipeline |
| **Week 4** | Streamlit app (form, what-if, batch), cold-start model, report and demo |

---

## 17. Limitations

- The dataset covers one UK retailer, so results may not transfer to other businesses.
- Churn is defined by a fixed 90-day window; a different window changes the labels.
- Customers without a `Customer ID` (guest checkouts) are excluded.
- Features typed manually in the form are approximations of the true history.
- The cold-start model is far less accurate than the full model because it has little information.
- Predictions show correlation, not causation: a high churn score does not prove a discount will retain the customer (see future enhancements).

---

## 18. Future Enhancements

- **Uplift modeling:** predict who responds to an offer, not just who will churn.
- **Survival analysis (Cox / DeepSurv):** predict *when* a customer will leave.
- **BG/NBD and Gamma-Gamma models** for probabilistic lifetime value.
- **Automated email or WhatsApp campaign trigger** for the priority list.
- **Monitoring dashboard** for data drift and model performance over time.
- **Fairness checks** across customer groups, such as country.

---

## 19. Expected Deliverables

1. Working Streamlit app with the customer check form, what-if simulator and batch upload
2. Clean, modular Python source code and saved models
3. Jupyter notebooks documenting each stage
4. Report with the model comparison table, SHAP plots and business impact analysis
5. Short demo video and presentation slides
6. GitHub repository with this README

---

## 20. References

- Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System* (KDD 2016)
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP, NeurIPS 2017)
- Chawla et al., *SMOTE: Synthetic Minority Over-sampling Technique* (JAIR 2002)
- Fader, Hardie & Lee, *Counting Your Customers the Easy Way* (BG/NBD model)
- UCI Machine Learning Repository: Online Retail II dataset
- Scikit-learn, XGBoost, SHAP and Streamlit documentation