# Results notes

## Known issues

- **Churn rate differs between snapshots.** Train (cutoff 2010-12-01) churn rate is 0.62; test
  (cutoff 2011-06-01) is 0.50. The train churn window (Dec-Feb) is the slow post-Christmas period.
  *Phase 5:* report the mean predicted churn probability on test next to the actual test churn rate,
  to see whether the probabilities are too high.
- **Spend-season mismatch.** The train spend window is Dec-May and the test spend window is Jun-Nov (includes
  the autumn peak). Mean future spend (active customers): train 794 GBP vs test 1,444 GBP.
  *Phase 6:* report this next to the spend model results; the model is expected to under-predict on test.
  *Phase 6 result:* confirmed. The final spend model (Linear Regression on log1p) predicts a mean of 645 GBP
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
