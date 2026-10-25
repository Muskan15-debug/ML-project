# Customer Retention Decision System: Churn Risk, Customer Value and Profit-Aware Campaign Planning

**Machine Learning Lab (20CP401P) - B.Tech Computer Engineering, Semester VII**
*Report draft. All numbers come from the saved result tables in `reports/tables/`.*
*Figure placeholders are written as [Figure X: caption - file name]; files are in `reports/figures/`.
Key figures are marked **(KEY)**.*

---

## Abstract

Retailers need to know which customers are about to leave and which of them are worth keeping. Using the Online Retail II
transaction log (1,067,371 rows, December 2009 - December 2011), we built leak-free customer-level datasets at two points
in time and trained models that predict churn (no purchase in the next 90 days) and 6-month future spend. Customers were
segmented with K-Means into Champions, Loyal, One-off buyers and At-Risk. Four classifiers were compared with three ways
of handling class imbalance; the final churn model is an XGBoost classifier with monotonic constraints, reaching ROC-AUC
0.757 and PR-AUC 0.730 on an out-of-time test set (recency-only baseline: 0.660). Future spend is predicted by a
two-part model (probability of buying x amount if buying), which beat a single log-regression in cross-validation
(RMSE 2,408 vs 2,830 GBP). Predictions are explained with SHAP and converted into actions with a discount-based profit
rule. A backtest on the test period shows that, at the default business assumptions, every campaign loses money and
the best decision is to run no campaign; mailing a discount to every customer would lose about 306,651 GBP. The value of
the models is that they identify the conditions under which targeting pays: with a small discount and a high response
rate, the profit rule becomes the best strategy. A Streamlit app makes predictions for typed-in customers and uploaded
CSV files.

## 1. Introduction and problem statement

Keeping an existing customer is usually cheaper than winning a new one, but sending offers to everyone wastes money on
customers who would have stayed anyway. A business therefore needs answers to two questions: *who is about to leave?*
and *who is worth keeping?* The Online Retail II dataset contains only raw transactions, so the labels, the features and
the decision rule all have to be built from scratch, without letting information from the future leak into the model.

## 2. Objectives

1. Clean and explore a large real transaction dataset.
2. Create churn and future-spend labels without data leakage.
3. Engineer customer-level RFM and behavioural features.
4. Segment customers with K-Means.
5. Select features with filter, wrapper and embedded methods.
6. Compare imbalance handling (none, class weights, SMOTE).
7. Train, tune and compare classifiers and regressors.
8. Explain predictions with SHAP and a readable decision tree.
9. Turn predictions into profit-based actions and backtest them.
10. Build an app that predicts for any customer from typed-in statistics.

## 3. Dataset

Online Retail II (UCI): a UK-based online gift retailer with many wholesale customers. Columns: invoice, stock code,
description, quantity, invoice date, price, customer ID, country. The data covers 2009-12-01 to 2011-12-09 with 5,942
customers and 53,628 invoices. Almost a quarter of the rows (243,007) have no customer ID (guest checkouts).

## 4. Methodology

### 4.1 Data cleaning and EDA (Phase 1)
Cancelled invoices (19,494 rows) were first saved in a separate table, because they are needed for the return-rate
feature, and then removed. We also removed bad-debt adjustments (6), rows without a customer ID (242,251), non-product
stock codes such as postage and fees (2,925), non-positive prices (61) and exact duplicates (26,055), leaving 776,579
rows, 5,852 customers and 36,594 orders. EDA showed a strong Christmas season, about 84% of revenue from the UK, a
median of 3 orders per customer, and very skewed spend (skewness 25.3, reduced to 0.27 by log1p).

[Figure 1: Monthly revenue, flat most of the year with a peak every November - monthly_revenue_matplotlib.png
(same chart in monthly_revenue_seaborn.png and monthly_revenue_plotly.png)]
[Figure 2: Top 10 countries by revenue; the UK dominates - top_countries_revenue.png]
[Figure 3: Top 10 products by units sold - top_products_quantity.png]
[Figure 4: Orders and total spend per customer on a log scale - customer_orders_spend_hist.png]
[Figure 5: Customer spend boxplots, raw and log1p - customer_spend_boxplot.png]

### 4.2 Labels, features and the out-of-time split (Phase 2)
For a cutoff date, features use only transactions before it and labels only transactions after it:
- churn = 1 if there is no purchase in the 90 days after the cutoff;
- future spend = total spend in the 6 months after the cutoff;
- only **active** customers (a purchase in the 180 days before the cutoff) are included.

Twelve features are built per customer: recency, frequency, monetary, average order value, average gap between orders,
tenure, number of different products, return rate (cancelled / (orders + cancelled) invoices), spend trend (share of
spend in the last 3 months), purchase regularity (spread of the gaps), a UK flag and a single-order flag. After
segmentation (4.3), the segment is added as 4 one-hot columns, so the full feature set has **12 + 4 = 16 columns**.
We compared candidate train cutoffs and chose **2010-12-01** for TRAIN (3,465 customers, churn rate 0.62) and
**2011-06-01** for TEST (2,658 customers, churn rate 0.50); the train label windows end exactly when the test snapshot
starts. Automated leakage tests (35 tests in total in the project) check that no feature uses a row on or after the
cutoff, that the label windows do not overlap the test snapshot, and that only active customers are included.

### 4.3 Customer segmentation (Phase 3)
Recency, log1p(frequency) and log1p(monetary) were standardised on TRAIN and clustered with K-Means. The elbow and
silhouette scores (0.383 for k=3, 0.345 for k=4) led to **k = 4**. Clusters were named by ranking their average values:
Champions, Loyal, One-off buyers and At-Risk. The segment was added as a one-hot feature.

[Figure 6: Elbow plot of K-Means inertia for k = 2-10 - kmeans_elbow.png]
[Figure 7: Silhouette score for k = 2-10 - kmeans_silhouette.png]
[Figure 8 (KEY): 2D PCA map of the four RFM segments - segments_pca_train.png]

### 4.4 Feature selection (Phase 4)
On TRAIN, skewed features were log1p-transformed and the segment one-hot encoded. Five methods ranked the features for
churn (mutual information, ANOVA F, RFE with Logistic Regression, Random Forest importance, L1-logistic) and four for
spend (mutual information, f_regression, Random Forest importance, Lasso). Each method voted for its top 8; the 8 most
voted features were kept for each task.

[Figure 9: Correlation between all 16 feature columns - feature_correlation_heatmap.png]
[Figure 10: Churn feature scores from four selection methods - feature_scores_churn.png]

### 4.5 Churn models and class imbalance (Phase 5)
Logistic Regression, CART, Random Forest and XGBoost were trained with no imbalance handling, class weights, and SMOTE
inside an imblearn Pipeline. Random Forest and XGBoost were tuned with a small GridSearchCV (stratified 5-fold, ROC-AUC).
The final model was chosen by **train cross-validation only**; TEST was used only for reporting.

[Figure 11 (KEY): ROC curves of the four churn models and the recency baseline on TEST - churn_roc_curves.png]
[Figure 12: Precision-recall curves on TEST - churn_pr_curves.png]
[Figure 13: Confusion matrices at threshold 0.5 on TEST - churn_confusion_matrices.png]
[Figure 14: Cumulative gain chart of the final churn model - churn_gain_chart.png]

### 4.6 Future spend models (Phases 6 and 6b)
Linear Regression, Random Forest and XGBoost were trained on log1p(future spend) and compared in GBP, together with a
historical-rate baseline and a median baseline. Duan's smearing correction was tested. Because about half of the
customers spend 0, a **two-part model** was then built: a classifier for P(spend > 0) and a regressor for the amount,
trained on buyers only (with smearing). Parts and the final model were chosen by train cross-validation in GBP.

[Figure 15: Predicted vs actual 6-month spend, single Linear Regression - spend_predicted_vs_actual.png]
[Figure 16 (KEY): Predicted vs actual 6-month spend, final two-part model - spend_predicted_vs_actual_two_part.png]

### 4.7 Explanations and business rules (Phase 7)
SHAP (TreeExplainer) explains the churn model globally and for single customers, shown with plain-English feature names;
a depth-3 CART tree gives readable rules. Each customer gets an action from a matrix of risk (high/low, split at the
TRAIN median churn probability 69.0%) and value (high/low, split at the TRAIN median expected spend 350 GBP). A discount
offer is kept only if the profit rule passes:
`expected_net = p x response x amount x (margin - discount) - (1 - p) x redeem x discount x amount - contact_cost > 0`,
where *amount* is the spend if the customer stays and buys. Five campaign strategies were backtested on TEST with the
real outcomes, and a sensitivity analysis covered 36 combinations of the assumptions.

[Figure 17 (KEY): SHAP summary - what pushes churn risk up or down - shap_summary_churn.png]
[Figure 18: SHAP explanation of one high-risk customer - shap_waterfall_high_risk.png]
[Figure 19: SHAP explanation of one low-risk customer - shap_waterfall_low_risk.png]
[Figure 20 (KEY): Net profit of each campaign strategy at the default assumptions - business_net_profit_by_strategy.png]

### 4.8 Saved pipeline, app and sanity checks (Phases 8, 9, 9b)
`src/train.py` rebuilds everything from the raw file and saves the segment model, encoder, feature lists, churn model,
spend model, decision thresholds and training medians. `src/predict.py` scores typed-in or uploaded customers with exactly
these files; the Streamlit app (`app.py`) has three tabs: Check a Customer (with What-If sliders), Batch Prediction and
a Dashboard with a live campaign simulator. A recency sanity check showed that the first XGBoost sometimes lowered risk
as recency increased, so the final model uses monotonic constraints.

[Figure 21: Churn probability vs recency for 4 typical customers, unconstrained XGBoost - recency_sanity_before.png]
[Figure 22 (KEY): The same check for the final monotonic model - recency_sanity_after.png]

## 5. Results

### 5.1 Segments

| Segment | Customers (train) | Avg recency | Avg orders | Avg spend (GBP) | Churn train | Churn test |
|---|---|---|---|---|---|---|
| Champions | 522 | 18.5 | 16.2 | 9,520 | 0.192 | 0.191 |
| Loyal | 1,229 | 32.8 | 4.5 | 1,679 | 0.554 | 0.485 |
| One-off buyers | 1,114 | 34.9 | 1.5 | 396 | 0.778 | 0.692 |
| At-Risk | 600 | 130.7 | 2.2 | 671 | 0.832 | 0.715 |

Churn rate clearly differs between segments, and the order holds on TEST (Figure 8).

### 5.2 Feature selection
Churn features: frequency, monetary, purchase_regularity, tenure_days, recency, seg_Champions, unique_products,
spend_trend. With Logistic Regression on TEST, the 8 selected features (ROC-AUC 0.764) perform about the same as all
16 columns, i.e. the 12 features plus the 4 one-hot segment columns (0.763). The segment adds nothing beyond RFM
(ROC-AUC 0.760 with and without it).

### 5.3 Churn models (TEST)

| Model | Imbalance | CV ROC-AUC (train) | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Recency only (baseline) | - | - | 0.657 | 0.660 | 0.642 |
| Logistic Regression | none | 0.772 | 0.721 | 0.764 | 0.742 |
| CART | none | 0.748 | 0.681 | 0.729 | 0.697 |
| Random Forest | none | 0.772 | 0.705 | 0.759 | 0.728 |
| XGBoost | none | 0.773 | 0.716 | 0.762 | 0.731 |
| **XGBoost, monotonic (final)** | none | 0.770 | 0.703 | **0.757** | **0.730** |

Imbalance handling (class weights or SMOTE) changed ROC-AUC by at most 0.006 for every model; the full 13-row table is
in `churn_model_comparison.csv`. All models beat the recency baseline by about 0.07-0.10 ROC-AUC; Logistic Regression,
Random Forest and XGBoost are practically tied (Figure 11). The final model's top 10% highest-risk customers contain
16.5% of all churners (lift 1.65); the top 20% contain 30.7%.

### 5.4 Spend models (TEST, GBP)

| Model | MAE | RMSE | R-squared | Mean predicted (actual 1,444) |
|---|---|---|---|---|
| Historical-rate baseline | 1,042 | 3,212 | 0.772 | 1,778 |
| Plain Linear Regression (log1p) | 1,006 | 4,169 | 0.616 | 645 |
| **Two-part model (final)** | **849** | 3,440 | 0.738 | **1,303** |

The two-part model clearly improves on the single regression (Figures 15 and 16), but the simple historical-rate
baseline still has the best TEST RMSE and R-squared, partly for seasonal reasons (it is the worst on TRAIN,
R-squared -0.263).

### 5.5 Explanations
Number of orders has the largest mean |SHAP| (0.667 log-odds), followed by days since first order (0.260), days since
last order (0.254) and total spent (0.244) (Figure 17). The readable CART tree (TEST ROC-AUC 0.735) starts with
"number of orders <= 4.5". After the monotonic constraint, predicted churn risk rises with recency for typical customers
(Figure 22).

## 6. Business impact

Default assumptions (not from data): margin 30%, discount 10%, response 20%, 100% of stayers redeem, 1 GBP per contact.

| Strategy | Contacted | Net profit on TEST (GBP) |
|---|---|---|
| Mail everyone | 2,658 | -306,651 |
| Top 20% highest churn risk | 532 | -915 |
| Profit rule | 809 | -7,547 |
| Priority offers only | 95 | -4,905 |
| **No campaign** | 0 | **0** |

**At the default assumptions every campaign loses money, and no campaign is the best decision** (Figure 20). Mailing
everyone would lose about 306,651 GBP. The reason is that the gain from a kept churner (20% response x 20% net margin,
about 4% of their spend) is smaller than the discount wasted on customers who stay (10% of their spend).

The value of the models is that they show **under which conditions targeting pays**. Across 36 combinations of the
assumptions (stayer redeem rate 30% / 50% / 100%, discount 5% / 10% / 20%, response 5% / 10% / 20% / 30%), no campaign
is best in 17, the **profit rule is the best strategy in 11**, the top 20% by risk in 6 and priority offers in 2;
mailing everyone never wins. The profit rule wins when the discount is small and the response is high: for example
+6,041 GBP at 5% discount, 20% response and 100% redeem, and its best result is **+57,244 GBP at 5% discount, 30%
response and 30% stayer redeem rate**.

The profit rule expected +6,146 GBP on TEST at the default assumptions but realised -7,547 GBP, because the
probabilities were too high (0.834 predicted vs 0.735 actual among the picked customers) and customers who stayed spent
more than predicted (905 vs 625 GBP).

## 7. Lab concepts used

| Lab topic | Used? | Where |
|---|---|---|
| Data visualization (Matplotlib, Seaborn, Plotly) | Yes | Notebook 01: monthly revenue in all three tools; all report figures |
| EDA workflow (cleaning, time aggregation, trends) | Yes | `data_cleaning.py`, notebook 01 |
| Simple / multiple linear regression | Yes | Spend model (Linear Regression on log1p spend; part 2 of the final two-part model) |
| Locally weighted regression | **No** | Skipped as agreed: tight timeline, and a one-feature curve adds little when half the targets are 0 |
| Feature selection (filter, wrapper, embedded) | Yes | Notebook 03: mutual information, ANOVA F, f_regression, RFE, RF importance, L1 / Lasso |
| Feature selection for classification AND regression | Yes | Separate churn and spend feature lists |
| Comparing classifiers | Yes | Notebook 04: Logistic Regression, CART, Random Forest, XGBoost |
| Oversampling imbalanced data (SMOTE) | Yes (did not help) | Notebook 04, inside imblearn pipelines; churners were the majority |
| CART decision tree | Yes | Classifier comparison + readable depth-3 rules |
| K-Means clustering | Yes | RFM segmentation, k by elbow + silhouette |
| PCA | Yes (visual only) | 2D map of the segments |
| Hyper-parameter tuning | Yes (small) | GridSearchCV (Optuna not used, as agreed, to keep tuning small and readable) |

Also **not built** from the original specification: the cold-start model for brand-new customers (an optional stretch
goal) and a separate `retrain.py` (`src/train.py` is the rerun script). Cross-validation inside TRAIN is stratified
random; the time-aware part of the evaluation is the separate out-of-time TEST snapshot.

## 8. Limitations

- One UK retailer; a fixed 90-day churn window; guest checkouts excluded.
- Season mismatch between the train and test windows: churn probabilities about 4-5 points too high on TEST, spend about
  10% too low.
- Moderate accuracy (churn ROC-AUC about 0.76) and rough individual spend predictions.
- Business results depend on assumed response, redeem and discount rates.
- Predictions show correlation, not causation: a high churn score does not prove a discount will keep the customer.
- Customers inactive for more than 180 days are outside the training population.

## 9. Future work

- Stack several cutoffs (for example 2010-06-01 and 2010-12-01) to cover all seasons.
- Calibrate probabilities on a recent period to remove the seasonal offset.
- Uplift modelling with real campaign data, to predict who responds to an offer rather than who churns.
- Survival analysis or BG/NBD + Gamma-Gamma models for timing and probabilistic lifetime value.
- A cold-start model for brand-new customers, and monitoring for data drift.

## 10. Work division

| Member | Responsibilities |
|---|---|
| Person A: [NAME, ROLL NO.] | Data cleaning and EDA, label creation and out-of-time split, feature engineering, K-Means segmentation, spend models (including the two-part model) |
| Person B: [NAME, ROLL NO.] | Churn models and imbalance handling, SHAP explanations and CART rules, business rules and campaign backtest, Streamlit app |

## 11. Conclusion

We built a complete, leak-free pipeline from raw transactions to an explained, profit-aware decision and a working app.
The churn models clearly beat a recency-only baseline but reach only moderate accuracy, and simple and complex models
perform almost the same. The most useful result is a business one: at the default assumptions no discount campaign
pays and mailing everyone would lose about 306,651 GBP, while the models identify the conditions (small discount, high
response) under which a targeted campaign chosen by the profit rule becomes the best strategy.

## 12. References

- Chen, T. and Guestrin, C. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- Lundberg, S. and Lee, S.-I. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
- Chawla, N. et al. *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR 2002.
- Duan, N. *Smearing Estimate: A Nonparametric Retransformation Method.* JASA 1983.
- Fader, P., Hardie, B. and Lee, K. *"Counting Your Customers" the Easy Way.* Marketing Science 2005.
- UCI Machine Learning Repository: Online Retail II dataset.
- scikit-learn, imbalanced-learn, XGBoost, SHAP and Streamlit documentation.
