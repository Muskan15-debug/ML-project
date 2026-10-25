# Viva questions and answers

Short, simple answers based on what we actually built. Numbers come from `reports/RESULTS_SUMMARY.md`.

**1. What does the project do?**
It looks at each customer's past purchases and predicts (a) whether they will buy nothing in the next 90 days (churn),
(b) how much they will spend in the next 6 months, and (c) whether a retention discount is worth sending. A Streamlit
app does this for one typed-in customer or a whole CSV, and explains each prediction.

**2. The dataset has no churn column. How did you create the label?**
We picked a cutoff date. Churn = 1 if the customer made no purchase in the 90 days after the cutoff. Future spend =
their total spend in the 6 months after the cutoff. Features are built only from purchases before the cutoff.

**3. How did you avoid data leakage?**
Three ways: (1) a strict time split: features use only rows dated before the cutoff, labels only rows on or after it;
(2) an out-of-time test: we train on the 2010-12-01 snapshot and test on the 2011-06-01 snapshot, and the train label
windows end exactly when the test snapshot begins; (3) scalers, K-Means, feature selection, encoders, SMOTE and all
thresholds are fitted on TRAIN only, and SMOTE runs inside an imblearn Pipeline so it only touches training folds.

**4. How do you know there is no leakage?**
We wrote 17 automated tests. The main one adds fake, huge purchases dated exactly on the cutoff and after it, and
checks that not a single feature value changes. Other tests check that features from all data equal features from data
already cut at the cutoff, that train label windows end before the test cutoff, and that recency and tenure are never
negative. Also, our best ROC-AUC is about 0.76: a leaking model usually looks "too good" (we had an alarm at 0.90).

**5. Why only active customers?**
Customers who last bought a year ago are almost certain "churners". Including them makes the task look easy and inflates
the churn rate, but a shop would never run a retention campaign for them. So we kept only customers who bought in the
180 days before the cutoff. After this, recency alone gives ROC-AUC about 0.66, so the models have to learn more.

**6. Why 90 days for churn and 6 months for spend?**
90 days is long enough that a regular customer of a gift wholesaler would normally have ordered again, and short enough
to act on. 6 months gives a stable value estimate and still fits inside the data (the test window ends on 2011-12-01,
the data ends on 2011-12-09). Both are settings in `config.py`; a different window would change the labels.

**7. Why is the train churn rate (0.62) higher than the test churn rate (0.50)?**
The train cutoff is 1 December, so its 90-day window is December-February, the slow period after Christmas. The test
window is June-August. Because of this, the model's probabilities run about 4-5 points high on the test data
(mean 0.541 vs 0.50). Ranking quality is not affected much: recency alone has ROC-AUC 0.661 on train and 0.660 on test.

**8. Why didn't SMOTE help?**
SMOTE creates synthetic examples of the minority class. Here churners are the MAJORITY (62% in train), so SMOTE made
synthetic non-churners. The classes were only moderately imbalanced, and ROC-AUC changed by at most 0.006 for every
model. SMOTE and class weights only pushed probabilities down and usually lowered F1 at the 0.5 threshold.

**9. Why XGBoost if Logistic Regression is just as good?**
Honestly, they tie: test ROC-AUC 0.757 for the final XGBoost and 0.764-0.767 for Logistic Regression. We chose by
cross-validation on train only (XGBoost 0.773 vs 0.772), never by test scores, to avoid tuning on test. XGBoost also
needs no scaling, captures non-linear effects, and works directly with SHAP's tree explainer. Logistic Regression would
be an equally defensible choice.

**10. What is the monotonic constraint and why did you add it?**
We checked how churn risk changes as "days since last order" grows. The first XGBoost sometimes gave a LOWER risk to a
customer who had been away longer, which makes no sense. A monotonic constraint forces the model so that risk can only
rise with recency and only fall with orders, total spent and tenure. It cost only 0.0023 train CV ROC-AUC
(0.7727 -> 0.7704), within our 0.005 limit, so we kept it.

**11. Why a two-part model for spend?**
About half of the customers spend 0 in the next 6 months, and a few spend over 100,000 GBP. One regression on
log(spend) squeezed both groups into one line and predicted far too little (645 vs 1,444 GBP on average). The two-part
model asks two simpler questions: "will they buy at all?" (Logistic Regression) and "how much if they do?" (Linear
Regression on buyers only). It had the lowest train CV error in GBP (RMSE 2,408 vs 2,830).

**12. How do you decide whether to send a discount?**
For each customer: expected_net = p x response x amount x (margin - discount) - (1 - p) x redeem x discount x amount
- contact cost. The first part is the profit from churners the offer keeps; the second is the discount wasted on
customers who would have stayed anyway; the third is the cost of contacting them. We send the offer only if
expected_net > 0. The amount is "what they would spend if they stay", so churn is not counted twice.

**13. Why does the "no campaign" strategy win at the default assumptions?**
With a 10% discount, a 20% response rate and the worst case that every customer who stays also uses the discount, a
kept churner earns back only about 4% of their spend (20% response x 20% net margin), while every stayer costs 10% of
their spend. So every campaign loses money on the test data. The key finding is the size of the loss avoided:
mailing everyone loses about 306,651 GBP, while model-based targeting loses at most about 7,547 GBP. Campaigns pay off
only when the discount is small and the response is high (e.g. +23,705 GBP at 5% discount and 30% response).

**14. What do the SHAP explanations show?**
Number of orders matters most (customers with many orders rarely leave), followed by days since first order, days
since last order and total spent. For one customer, SHAP shows which values pushed their risk up or down, for example
"179 days since last order -> raises risk". The app shows the top reasons in plain English.

**15. How does the app use the same model as training, and what happens with unusual input?**
The app only loads the saved files created by `src/train.py` (segments, encoder, feature list, churn model, two-part
spend model, thresholds, medians) and calls the same feature rules. A test checks that, for 300 test customers, the
app's probabilities equal the training pipeline's within 0.001. Impossible inputs (e.g. last-3-months spend above total
spend) are rejected with a clear message; customers inactive for more than 180 days still get a result, with a
warning that they are outside the training population.
