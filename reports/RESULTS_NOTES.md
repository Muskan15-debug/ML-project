# Results notes

## Known issues

- **Churn rate differs between snapshots.** Train (cutoff 2010-12-01) churn rate is 0.62; test
  (cutoff 2011-06-01) is 0.50. The train churn window (Dec-Feb) is the slow post-Christmas period.
  *Phase 5:* report the mean predicted churn probability on test next to the actual test churn rate,
  to see whether the probabilities are too high.
- **Spend-season mismatch.** The train spend window is Dec-May and the test spend window is Jun-Nov (includes
  the autumn peak). Mean future spend (active customers): train 794 GBP vs test 1,444 GBP.
  *Phase 6:* report this next to the spend model results; the model is expected to under-predict on test.
- **Possible later improvement (not implemented):** stack the 2010-06-01 and 2010-12-01 snapshots as one
  training set to cover both seasons. This would need customer-grouped cross-validation.
- **Bar to beat.** Recency alone gives ROC-AUC about 0.66 for churn (0.661 train, 0.660 test).
  *Phase 5:* include this recency-only baseline in the model comparison table.
