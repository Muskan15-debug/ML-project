# Results notes

## Main business finding

Backtest on the test snapshot (2,658 active customers, June-November 2011), with the final (monotonic) churn
model. The offer is a discount; margin, discount, response rate, stayer redeem rate and contact cost are
ASSUMPTIONS (defaults: 30% margin, 10% discount, 20% response, 100% of stayers redeem, 1 GBP per contact).

- **Mailing everyone loses heavily.** At the defaults it would lose about 306,651 GBP, mostly discount given to
  customers who would have stayed anyway. Across all 36 tested assumption combinations it never wins and loses up to
  about 694,000 GBP.
- **Model-based targeting avoids most of that loss.** Contacting only the customers the models pick keeps the loss
  small at the defaults (top 20% by risk -915 GBP, profit rule -7,547 GBP, priority offers -4,905 GBP), so about 97%
  or more of the "mail everyone" loss is avoided.
- **Discount campaigns only pay off when the discount is small and the response is high.** The profit rule makes money
  with a 5% discount once 10-20% or more of churners respond (best case +57,244 GBP), and with a 10% discount only when
  few stayers redeem it and response is high. With a 20% discount almost nothing pays.
- **The system can tell the business when NOT to run a campaign.** At the defaults, and in 17 of 36 combinations,
  the best choice is no campaign at all.

Why the profit rule underperformed its own estimate on test (it expected +6,146 GBP for its 809 customers but got
-7,547 GBP):
1. **Churn probabilities are too high:** the customers it picked had a mean predicted churn probability of 0.834, but
   only 0.735 really churned, so more discounts went to customers who stayed.
2. **Stayers spend more than predicted:** they spent 905 GBP on average vs 625 GBP predicted (season mismatch), so
   each wasted discount cost more than planned.

## Known issues

- **Churn rate differs between snapshots.** Train (cutoff 2010-12-01) churn rate is 0.62; test
  (cutoff 2011-06-01) is 0.50. The train churn window (Dec-Feb) is the slow post-Christmas period.
  *Phase 5:* report the mean predicted churn probability on test next to the actual test churn rate,
  to see whether the probabilities are too high.
- **Spend-season mismatch.** The train spend window is Dec-May and the test spend window is Jun-Nov (includes
  the autumn peak). Mean future spend (active customers): train 794 GBP vs test 1,444 GBP.
  *Phase 6:* report this next to the spend model results; the model is expected to under-predict on test.
  *Phase 6 result:* confirmed. The single spend model (Linear Regression on log1p) predicts a mean of 645 GBP
  on test vs an actual 1,444 GBP. Part of this gap is the season mismatch and part is the log-transform bias.
  Duan's smearing (factor about 18, learned on train) did **not** help: it over-corrected to a mean of about
  6,700 GBP because about half the customers spend 0, so the log errors are huge. Smearing could only ever fix
  the log-transform bias, never the season mismatch.
  *Phase 6b result (two-part model):* the spend model is now a two-part model: Logistic Regression for
  P(spend > 0) x Linear Regression amount (trained on spenders only, smearing factor 1.65). It replaced the
  single model because its train CV RMSE in GBP is lower (2,408 vs 2,830). The season mismatch **still causes
  under-prediction**: on train CV it predicts 21% too high (963 vs 794 GBP), but on test 10% too low
  (1,303 vs 1,444 GBP). The two-part model never predicts exactly 0, so customers who will not buy still get a
  few hundred GBP of expected value.
- **Historical-rate baseline beats every spend model on test in GBP.** Baseline = monetary / max(tenure, 30) * 182.
  Test: R-squared 0.772 / RMSE 3,212 GBP vs Linear Regression 0.616 / 4,169 GBP. But on train CV the baseline is the
  worst (R-squared -0.263 vs 0.364 for Linear Regression). Its test win is partly seasonal: the customer's past
  history includes the Christmas peak, and so does the test window (Jun-Nov), but the train window (Dec-May) does not.
  The models rank customers better on the log scale (R-squared 0.26 vs -0.23).
  *After Phase 6b:* the two-part model narrows the gap: on test it now has a lower MAE than the baseline
  (849 vs 1,042 GBP; spenders only 1,018 vs 1,055), but the baseline still has the best RMSE / R-squared
  on test (3,212 / 0.772 vs 3,440 / 0.738).
- **Possible later improvement (not implemented):** stack the 2010-06-01 and 2010-12-01 snapshots as one
  training set to cover both seasons. This would need customer-grouped cross-validation.
- **Bar to beat.** Recency alone gives ROC-AUC about 0.66 for churn (0.661 train, 0.660 test).
  *Phase 5:* include this recency-only baseline in the model comparison table.
- **Old flat-cost profit rule replaced (Phase 7).** With a flat 5 GBP offer cost, every customer's expected loss was
  above the cost, so the rule equalled "mail everyone". It was replaced by a discount cost model:
  expected_net = p x response x amount x (margin - discount) - (1 - p) x stayer_redeem x discount x amount - contact_cost,
  send only if expected_net > 0. All five inputs are ASSUMPTIONS (margin 30%, discount 10%, contact 1 GBP,
  response 20%, stayer redeem 100% = worst case).
- **Backtest result: at the default assumptions, no campaign is best.** On test every strategy loses money:
  mail everyone -306,651 GBP, profit rule -7,547, priority offers -4,905, top 20% by risk -915, no campaign 0.
  Over the 12 combinations of response rate (5-30%) and discount (5-20%) at 100% stayer redeem, "no campaign" is best
  in 7; the profit rule wins only with a 5% discount and 20-30% response (+6,041 / +23,705 GBP).
- **The profit rule is too optimistic on test.** It expected +6,146 GBP for its 809 customers but got -7,547 GBP:
  their mean predicted churn probability was 0.834 vs 0.735 actual, and stayers spent 905 GBP on average vs 625
  predicted (season mismatch), so the wasted discounts cost more than planned. The 100% stayer redeem rate is a worst
  case; a real rate below 100% would make campaigns look better.
- **Churn probabilities on test:** the final churn model predicts a mean of 0.541 vs an actual test churn rate
  of 0.50 (the unconstrained model predicted 0.550). The risk split (TRAIN median 69.0%) therefore puts fewer test
  customers in "high risk".
- **Recency sanity check (Phase 9b).** The unconstrained XGBoost sometimes predicted LOWER risk for a customer who had
  been away longer (34 downward steps on a 5-day grid, up to -0.18). The final model uses monotonic constraints
  (risk rises with recency, falls with orders, total spent and tenure). Train CV ROC-AUC 0.7704 vs 0.7727 (within the
  agreed 0.005), test ROC-AUC 0.757 vs 0.762. 7 tiny dips (at most 0.0095) remain, only for one-order customers,
  where tenure and recency move together.
