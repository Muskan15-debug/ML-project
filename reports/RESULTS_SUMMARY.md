# Results summary - Customer Lifetime Value & Churn Predictor

All numbers below come from the saved files in `reports/tables/`, `models/` and the notebook outputs.
TRAIN = snapshot at cutoff 2010-12-01, TEST = snapshot at cutoff 2011-06-01 (out-of-time).
GBP values in the business section depend on ASSUMPTIONS (margin, discount, response, redeem rate, contact cost).

---

## 1. Dataset and cleaning

Online Retail II (UCI): a UK online gift retailer, 1 December 2009 to 9 December 2011.

| | Rows | Customers | Invoices |
|---|---|---|---|
| Raw data | 1,067,371 | 5,942 | 53,628 |
| After cleaning | 776,579 | 5,852 | 36,594 |

Rows removed, in order (`cleaning_log.csv`):

| Step | Rows removed |
|---|---|
| Cancelled invoices (start with "C") - kept separately for `return_rate` | 19,494 |
| Bad-debt adjustment invoices (start with "A") | 6 |
| Missing customer ID (guest checkouts) | 242,251 |
| Non-product stock codes (postage, fees, adjustments, tests...) | 2,925 |
| Quantity <= 0 | 0 |
| Price <= 0 | 61 |
| Exact duplicate rows | 26,055 |

Clean revenue: 17,068,583 GBP. Main EDA findings: strong Christmas season (monthly revenue peaks about 1.16M GBP in
Nov 2010 and 1.14M GBP in Nov 2011), the UK brings about 84% of revenue, the median customer has 3 orders and 28% ordered
only once, the top 10% of customers bring 64% of revenue, and customer spend has skewness 25.3 (0.27 after log1p).

## 2. Labels and out-of-time split

- **Churn = 1** if the customer makes **no purchase in the 90 days after the cutoff**.
- **Future spend** = total spend in the **6 months after the cutoff**.
- **Features** use only transactions **before** the cutoff (checked by 17 leakage tests).
- **Customer base = active customers only**: at least one purchase in the 180 days before the cutoff.

Cutoff comparison, active customers only (`cutoff_comparison_active.csv`):

| Role | Cutoff | History (months) | Customers | Churn rate | Mean future spend | Recency-only ROC-AUC |
|---|---|---|---|---|---|---|
| candidate | 2010-06-01 | 6.0 | 2,652 | 0.50 | 1,373.3 | 0.649 |
| candidate | 2010-09-01 | 9.0 | 2,822 | 0.38 | 1,254.9 | 0.636 |
| **TRAIN** | **2010-12-01** | 12.0 | **3,465** | **0.62** | 793.6 | 0.661 |
| **TEST** | **2011-06-01** | 18.0 | **2,658** | **0.50** | 1,444.3 | 0.660 |

Share of customers with future spend = 0: 46.1% in TRAIN, 29.8% in TEST.

## 3. Key decisions and why (in simple words)

1. **Active customers only (bought in the last 180 days).** Without this filter, customers who left a year ago were
   counted as "churners". They are easy to predict and inflate the churn rate, but nobody would run a retention
   campaign for them. With the filter, recency alone reaches only ROC-AUC about 0.66, so the models must really learn.
2. **Out-of-time split, train cutoff 2010-12-01.** We train on one point in time and test on a later one, like real use.
   The train label windows end on 2011-06-01, exactly when the test snapshot starts, so they never overlap.
   2010-12-01 gave the most history (12 months) and the most customers. Known side effect: its churn window is the
   slow post-Christmas period, so TRAIN churn (0.62) is higher than TEST churn (0.50).
3. **SMOTE did not help.** In TRAIN, churners are the **majority** (62%). SMOTE and class weights therefore boost the
   non-churners. Ranking quality (ROC-AUC) moved by at most 0.006 for every model; they only shifted the probabilities
   down and usually lowered F1 at the 0.5 threshold.
4. **XGBoost and Logistic Regression tie.** XGBoost had the best train CV ROC-AUC (0.773 vs 0.772), but on TEST
   Logistic Regression is equal or slightly better (0.764-0.767 vs 0.762). XGBoost was kept because it won in CV
   (chosen without looking at TEST) and works directly with SHAP's tree explainer.
5. **Monotonic constraints (Phase 9b).** The unconstrained XGBoost sometimes gave LOWER risk to a customer who had been
   away longer (34 downward steps on a 5-day recency grid, the largest -0.18). We added constraints (risk rises with
   recency, falls with orders, total spent and tenure). Train CV ROC-AUC dropped only 0.0023 (0.7727 -> 0.7704), inside
   the agreed 0.005 limit, so the constrained model became the final churn model. 7 tiny dips (at most 0.0095) remain,
   only for one-order customers, where tenure and recency move together.
6. **Two-part spend model replaced plain regression.** About half of the customers spend 0. A single regression on
   log1p(spend) predicted far too little (mean 645 vs 1,444 GBP on TEST). The two-part model predicts
   P(spend > 0) x amount-if-they-buy and has a lower train CV RMSE in GBP (2,408 vs 2,830), so it replaced the single model.
7. **Why smearing failed for the single model.** Duan's smearing corrects the log-transform bias by multiplying by the
   average of exp(error). With half the customers at exactly 0, the log errors were huge (CV RMSE about 2.7), the factor
   became about 18 and predictions exploded (mean about 6,700 GBP). Inside the two-part model, trained on spenders only,
   the factor is a sensible 1.65.
8. **The profit rule uses the conditional amount.** The value at risk is what the customer would spend *if they stay
   and buy*. Using expected spend (P(spend > 0) x amount) would count churn twice, because likely churners already have
   a low P(spend > 0).
9. **Discount-based profit rule.** A flat 5 GBP offer cost made every customer look worth contacting. The real cost of
   a discount is that customers who would have stayed anyway also use it. The final rule:
   `expected_net = p x response x amount x (margin - discount) - (1 - p) x redeem x discount x amount - contact_cost`,
   send only if `expected_net > 0`.
10. **"No campaign" can be the right answer.** At the default assumptions every campaign loses money on TEST, mailing
    everyone loses about 306,651 GBP, and model-based targeting avoids about 97% or more of that loss.
11. **Reproducibility.** `python src/train.py` rebuilt everything from the raw CSV alone (94 s) and reproduced the saved
    tables and models exactly; after the monotonic change it was re-run again (121 s) and matched notebook 04
    (CV 0.7704, TEST ROC-AUC 0.7567). 35 automated tests pass (leakage and prediction).

## 4. Customer segments (K-Means, k = 4)

k = 4 was chosen from the elbow (inertia 4,066 at k=3, 3,128 at k=4, 2,677 at k=5) and silhouette
(0.383 at k=3, 0.345 at k=4). k = 4 separates regular buyers from one-off buyers, which churn very differently.

| Segment | Customers (train) | Avg recency (days) | Avg orders | Avg spend (GBP) | Churn train | Churn test | Meaning |
|---|---|---|---|---|---|---|---|
| Champions | 522 | 18.5 | 16.2 | 9,520 | 0.192 | 0.191 | Buy recently, often and the most; rarely leave |
| Loyal | 1,229 | 32.8 | 4.5 | 1,679 | 0.554 | 0.485 | Regular buyers; about half leave |
| One-off buyers | 1,114 | 34.9 | 1.5 | 396 | 0.778 | 0.692 | Recent but only 1-2 orders; most do not return |
| At-Risk | 600 | 130.7 | 2.2 | 671 | 0.832 | 0.715 | Still active but quiet for about 4 months |

## 5. Feature selection

Five churn methods (mutual information, ANOVA F, RFE with Logistic Regression, Random Forest importance, L1-logistic)
each voted for their top 8 features. Chosen churn features: **frequency, monetary, purchase_regularity, tenure_days,
recency, seg_Champions, unique_products, spend_trend**. Spend features: monetary, purchase_regularity, unique_products,
frequency, seg_Champions, avg_gap_days, avg_order_value, tenure_days.
Highly correlated pairs (|r| > 0.8): recency / seg_At-Risk (0.829) and frequency / monetary (0.820).

| Logistic Regression on TEST | Features | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Recency only (baseline) | 1 | 0.657 | 0.660 | 0.642 |
| All features | 16 | 0.707 | 0.763 | 0.745 |
| Selected features | 8 | 0.721 | 0.764 | 0.742 |

The 8 selected features perform **about the same** as all 16. The segment adds nothing beyond RFM
(ROC-AUC 0.760 for RFM only and for RFM + segment).

## 6. Churn models (TEST, threshold 0.5, actual churn rate 0.50)

| Model | Imbalance | CV ROC-AUC (train) | Precision | Recall | F1 | ROC-AUC | PR-AUC | Mean pred. prob. |
|---|---|---|---|---|---|---|---|---|
| Recency only (baseline) | - | - | 0.607 | 0.714 | 0.657 | 0.660 | 0.642 | - |
| Logistic Regression | none | 0.772 | 0.655 | 0.802 | 0.721 | 0.764 | 0.742 | 0.548 |
| Logistic Regression | class weights | 0.772 | 0.710 | 0.677 | 0.693 | 0.765 | 0.742 | 0.459 |
| Logistic Regression | SMOTE | 0.771 | 0.704 | 0.692 | 0.698 | 0.767 | 0.744 | 0.468 |
| CART | none | 0.748 | 0.627 | 0.745 | 0.681 | 0.729 | 0.697 | 0.568 |
| CART | class weights | 0.744 | 0.701 | 0.616 | 0.656 | 0.729 | 0.696 | 0.477 |
| CART | SMOTE | 0.755 | 0.650 | 0.693 | 0.671 | 0.733 | 0.701 | 0.509 |
| Random Forest | none | 0.772 | 0.667 | 0.748 | 0.705 | 0.759 | 0.728 | 0.549 |
| Random Forest | class weights | 0.772 | 0.711 | 0.629 | 0.667 | 0.761 | 0.729 | 0.463 |
| Random Forest | SMOTE | 0.771 | 0.713 | 0.638 | 0.674 | 0.765 | 0.739 | 0.468 |
| XGBoost | none | 0.773 | 0.665 | 0.774 | 0.716 | 0.762 | 0.731 | 0.550 |
| XGBoost | class weights | 0.772 | 0.709 | 0.619 | 0.661 | 0.759 | 0.733 | 0.457 |
| XGBoost | SMOTE | 0.772 | 0.702 | 0.655 | 0.678 | 0.764 | 0.739 | 0.476 |
| **XGBoost, monotonic (FINAL)** | none | **0.770** | 0.670 | 0.739 | 0.703 | **0.757** | 0.730 | 0.541 |

Final model settings: learning rate 0.03, max depth 3, 200 trees. Lift (final model): the 10% highest-risk customers
contain 16.5% of all churners (lift 1.65); the top 20% contain 30.7% (lift 1.53). Because half of the test customers
churn, the best possible lift is 2.0.

## 7. Spend models (6-month future spend, GBP)

TRAIN cross-validation (used for the choice):

| Model | CV RMSE (GBP) | CV MAE | CV R-squared | CV mean predicted (actual 794) |
|---|---|---|---|---|
| Historical-rate baseline | 3,990 | 1,374 | -0.263 | 1,981 |
| Linear Regression (log1p) | 2,830 | 622 | 0.364 | 425 |
| Random Forest (log1p) | 3,154 | 623 | 0.210 | 268 |
| XGBoost (log1p) | 3,174 | 626 | 0.201 | 245 |
| Linear Regression + smearing | 61,436 | 7,106 | -298.494 | 7,668 |
| **Two-part: Logistic + Linear (FINAL)** | **2,408** | 623 | **0.540** | 963 |

TEST results (reported only, not used to choose):

| Model | MAE | RMSE | R-squared | MAE spenders only | Mean predicted (actual 1,444) |
|---|---|---|---|---|---|
| Historical-rate baseline | 1,042 | 3,212 | 0.772 | 1,055 | 1,778 |
| Median baseline | 1,397 | 6,853 | -0.038 | 1,935 | 128 |
| Plain Linear Regression | 1,006 | 4,169 | 0.616 | 1,397 | 645 |
| **Two-part model (FINAL)** | **849** | 3,440 | 0.738 | **1,018** | **1,303** |

Honest reading: the two-part model is clearly better than the single model, but individual errors are still large
(MAE about 850 GBP vs a median test spend of 407 GBP), and the simple historical-rate baseline has the best TEST RMSE
and R-squared, partly because its history contains the Christmas peak, like the test window (it is the worst on TRAIN).

## 8. What drives churn risk (SHAP, final model, 300 test customers)

| Feature (plain English) | Mean abs. SHAP (log-odds) | Effect |
|---|---|---|
| Number of orders | 0.667 | More orders -> much lower risk |
| Days since first order | 0.260 | Long-standing customers are safer |
| Days since last order | 0.254 | Longer gap -> higher risk |
| Total spent | 0.244 | More spent -> lower risk |
| Share of spend in last 3 months | 0.148 | Low recent share -> higher risk |
| Variation in days between orders | 0.148 | Irregular buyers -> higher risk |
| Different products bought | 0.072 | Small effect |
| In Champions segment | 0.053 | Small effect |

Readable CART tree (depth 3, TEST ROC-AUC 0.735): the first rule is "number of orders <= 4.5"; customers with few
orders mostly churn.

## 9. Business impact (TEST backtest)

Default ASSUMPTIONS: gross margin 30%, discount 10%, response rate 20%, 100% of stayers redeem, 1 GBP per contact.

| Strategy | Contacted | Churners reached | Wasted discount (GBP) | Net profit (GBP) |
|---|---|---|---|---|
| (a) Mail everyone | 2,658 | 1,330 | 348,394 | -306,651 |
| (b) Top 20% highest churn risk | 532 | 407 | 8,511 | -915 |
| (c) Profit rule (expected_net > 0) | 809 | 595 | 19,367 | -7,547 |
| (d) Priority offers only | 95 | 60 | 7,888 | -4,905 |
| **(e) No campaign** | 0 | 0 | 0 | **0** |

Sensitivity at 100% stayer redeem (best strategy and its net profit in GBP):

| Discount \ Response | 5% | 10% | 20% | 30% |
|---|---|---|---|---|
| 5% | No campaign | Top 20% (+292) | Profit rule (+6,041) | Profit rule (+23,705) |
| 10% | No campaign | No campaign | No campaign | Top 20% (+3,149) |
| 20% | No campaign | No campaign | No campaign | Priority offers (+150) |

Over all 36 combinations (redeem 30% / 50% / 100% x discount x response): no campaign is best in 17, the profit rule in
11 (best case +57,244 GBP at 30% redeem, 5% discount, 30% response), top 20% in 6, priority offers in 2. Mailing everyone
never wins and loses up to 693,896 GBP.

The profit rule expected +6,146 GBP for its 809 customers but got -7,547 GBP, because (1) their mean predicted churn
probability was 0.834 while only 0.735 churned, and (2) stayers spent 905 GBP on average vs 625 GBP predicted.

## 10. Lab concepts used

| Lab topic | Used? | Where |
|---|---|---|
| Data visualization (Matplotlib, Seaborn, Plotly) | Yes | Notebook 01: monthly revenue in all three tools; all report figures; Plotly-ready app |
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
| Hyper-parameter tuning | Yes (small) | GridSearchCV (Optuna not used, as agreed) |

Also **not built** from the original spec: the cold-start model (optional stretch goal) and a separate `retrain.py`
(`src/train.py` is the rerun script). Cross-validation inside TRAIN is stratified random; the time-aware part is the
out-of-time TEST snapshot.

## 11. Limitations

From the spec:
- One UK retailer only; results may not transfer to other businesses.
- Churn depends on the fixed 90-day window; another window would change the labels.
- Guest checkouts (242,251 rows without a customer ID) are excluded.
- Values typed into the app are approximations of the true history (purchase regularity is assumed for 3+ orders).
- Predictions show correlation, not causation: a high churn score does not prove that a discount will keep the customer.

Discovered during the project:
- **Season effects:** TRAIN churn 0.62 vs TEST 0.50, and TRAIN spend window Dec-May vs TEST Jun-Nov. Churn probabilities
  run about 4-5 points high on TEST (0.541 vs 0.50) and spend is under-predicted by about 10%.
- **Moderate accuracy:** churn ROC-AUC about 0.76; all good models tie, so the limit is the information in the data.
- **Spend predictions are rough** for individuals, and a simple baseline wins on TEST RMSE.
- **Business numbers depend on assumptions** (response rate, redeem rate, discount) that are not in the data.
- **Active-customer scope:** customers away for more than 180 days are outside the training population (the app warns).
- **Monotonic constraints** keep risk sensible in each feature separately, not when several features change together.
- Only two snapshots are used; stacking more cutoffs could cover more seasons.

## 12. Figures (`reports/figures/`)

| File | Caption |
|---|---|
| monthly_revenue_matplotlib.png | Monthly revenue (Matplotlib): flat most of the year, peaks every November |
| monthly_revenue_seaborn.png | The same monthly revenue chart in Seaborn |
| monthly_revenue_plotly.png | The same monthly revenue chart in Plotly |
| top_countries_revenue.png | Top 10 countries by revenue; the UK dominates (about 84%) |
| top_products_quantity.png | Top 10 products by units sold: cheap gift and party items |
| customer_orders_spend_hist.png | Orders and total spend per customer (log scale): very skewed |
| customer_spend_boxplot.png | Customer spend boxplots, raw and log1p: big outliers, log fixes the skew |
| kmeans_elbow.png | Elbow plot of K-Means inertia for k = 2-10 |
| kmeans_silhouette.png | Silhouette score for k = 2-10 (best k = 3, k = 4 close) |
| segments_pca_train.png | 2D PCA map of the four RFM segments (TRAIN) |
| feature_correlation_heatmap.png | Correlation between all features; two pairs above 0.8 |
| feature_scores_churn.png | Churn feature scores from four selection methods |
| churn_roc_curves.png | ROC curves of the four churn models and the recency baseline (TEST) |
| churn_pr_curves.png | Precision-recall curves of the four churn models and the recency baseline (TEST) |
| churn_confusion_matrices.png | Confusion matrices of the four churn models at threshold 0.5 (TEST) |
| churn_gain_chart.png | Cumulative gain chart of the final churn model (TEST) |
| recency_sanity_before.png | Churn probability vs recency for 4 typical customers, unconstrained XGBoost |
| recency_sanity_after.png | The same check for the final monotonic model |
| spend_predicted_vs_actual.png | Predicted vs actual 6-month spend, single Linear Regression (TEST, log scale) |
| spend_predicted_vs_actual_two_part.png | Predicted vs actual 6-month spend, final two-part model (TEST, log scale) |
| shap_summary_churn.png | SHAP summary: what pushes churn risk up or down (final model) |
| shap_waterfall_high_risk.png | SHAP explanation of one high-risk customer |
| shap_waterfall_low_risk.png | SHAP explanation of one low-risk customer |
| business_net_profit_by_strategy.png | Net profit of each campaign strategy at the default assumptions (TEST) |

Viva questions and answers: see `reports/VIVA_QA.md`. Report draft: `reports/REPORT_DRAFT.md`.
