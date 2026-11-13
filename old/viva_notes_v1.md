# Viva Notes: NIFTY 50 Direction Prediction

Quick revision sheet. For every technique: **What** it is, **Why** we used it, and what we **Got**.

---

## The problem in one line
Predict if tomorrow's NIFTY 50 close is higher (1) or lower/equal (0) than today's. This is **binary classification**.

**Key rule:** the prediction is made at **3:30 PM IST on day t**, so features may only use information known at that time.

---

## 1. EDA
| Finding | Number | What we did about it |
|---|---|---|
| No missing values, no duplicate dates | 2,912 days | nothing needed |
| Zero-volume days (data errors) | 28 days | replaced with previous day's volume |
| Price is not stationary (7k → 26k) | - | used returns/ratios, not raw price |
| Fat tails | kurtosis ≈ 17, skew ≈ -1, worst day -13% (23 Mar 2020) | robust tree models, clipped crude oil |
| Volatility clustering | autocorr of \|returns\| 0.2-0.3 | added volatility features |
| Direction is nearly random | autocorr of returns ≈ 0 | expect accuracy only slightly above 50% |
| Mild class imbalance | 53.3% UP overall; train 55.2%, test 48.3% | ROC-AUC + "always UP" baseline |

## 2. Feature Derivation
- **31 features** in total: returns (1/5/10 days), lags (1-4), trend (close vs SMA 10/20/50/200), momentum (RSI, MACD, Stochastic), volatility (Bollinger width, ATR, 20-day std), candle (gap, intraday return, close position), volume (volume ratio, OBV), 52-week high distance, day of week, India VIX, Bank Nifty, S&P 500, Nasdaq, USD/INR, crude.
- **Why ratios and returns:** they mean the same thing in 2016 and in 2026, while raw prices do not.
- **Why global markets:** India reacts to the US, oil and the rupee. We use only **yesterday's** foreign values, because those markets close after 3:30 PM IST.
- **Removed 5 near-duplicates** (|corr| > 0.9, measured on training data only): `close_vs_sma10`≈`ret_5`, `close_vs_sma20`≈`ret_10`, `volatility_20`≈`atr_pct`, `vix`≈`atr_pct`, `nasdaq_ret`≈`sp500_ret`. **26 remain.**
- Every feature has |correlation| < 0.07 with tomorrow's return, so each signal is weak on its own.

## 3. Classification (Logistic Regression)
- **What:** weighted sum of features → sigmoid → probability of UP. Weights are learned by minimising log-loss.
- **Why:** standard baseline; gives probabilities; interpretable weights; hard to overfit.
- **Scaling (StandardScaler)** is needed so that big-number features (RSI, 0-100) do not dominate small ones (returns, ~0.01).
- **Got:** train 56.3%, test 47.9%. Even a linear model overfits → regularization is needed.

## 4. Regularization
- **What:** add a penalty for complexity to the loss. `C = 1/strength`, so small C = strong regularization.
  - **L1 (Lasso)**, Σ|w|: sets useless weights to exactly 0, so it also selects features.
  - **L2 (Ridge)**, Σw²: shrinks all weights but keeps every feature.
  - **Trees:** `max_depth`, `min_child_samples` / `min_samples_leaf`, `subsample`, `colsample_bytree`, `reg_lambda`, small `learning_rate`.
- **Why:** stock data is mostly noise, so an unregularized model memorises the noise.
- **Got:**
  - L2 at the strongest setting (C = 0.001) was best (validation AUC 0.513). Bigger C → train AUC up, validation down: overfitting.
  - L1 at C = 0.03 kept only 5 features (`vix_change`, `close_vs_sma50`, `sp500_ret`, `stoch_k`, `close_position`). At C ≤ 0.01 it zeroes everything (underfitting).
  - L2 > L1 here because the signal is spread thinly over many features.
  - LightGBM: train AUC 1.000 / validation 0.517 without regularization → 0.802 / 0.520 with it (gap 0.48 → 0.28).

## 5. Hyperparameter Tuning
- **What:** GridSearchCV tries all combinations; scored by ROC-AUC with **TimeSeriesSplit (5 folds)**.
- **Why TimeSeriesSplit:** normal K-fold shuffles, so the model would train on the future and test on the past (leakage, inflated score).
- **Why AUC:** accuracy can look good just by always saying UP; AUC measures ranking quality, independent of the threshold.
- **Got (validation AUC):** LR 0.513 · SVM 0.530 · **Random Forest 0.543** (depth 5, min leaf 100) · XGBoost 0.528 · LightGBM 0.528 (both depth 2, lr 0.01).
- **Pattern:** every model picked its most conservative settings, so simpler = better on this data.

## 6. Ensembles
| Type | Model | Reduces | Validation AUC |
|---|---|---|---|
| Bagging | Random Forest | variance | **0.543** (best) |
| Boosting | XGBoost / LightGBM | bias | 0.528 |
| Soft Voting | LR + RF + XGB + LGBM | - | 0.529 |
| Stacking | same 4 + LR meta-model | - | 0.530 |

- Bagging wins because noisy data needs **variance reduction**. Boosting's strength (fitting harder) mostly fits noise here.
- The stacking meta-model gave LR ≈ 0 weight and trusted the tree models, so it learned which models are useful.

## 7. Performance Analysis
- **Final model = Random Forest**, chosen by **validation** AUC (choosing by test score would be cheating).
- **Test (Sep 2024 - Oct 2026, 532 days):** accuracy 51.7%, AUC 0.520, precision 0.50, recall 0.83, F1 0.62.
- **Baselines (test):** always UP 48.3%, same as today 52.4%.
- **Overfitting:** SVM train 86% vs test 48% (huge gap). RF gap 9 points.
- **Confusion matrix:** the model says UP on ~80% of days; it learned an UP bias from a bull market (55.2% UP) and was tested in a falling market (48.3% UP) → regime change.
- **Backtest (0.05% cost per trade):** buy-and-hold -8.3%, RF strategy -6.5%, same max drawdown. It stayed invested on 99% of days, so the difference is likely luck.
- **7-year walk-forward (2019-2026):** RF 53.9% vs always UP 53.6% vs same as today 54.6%. **No clear edge over the baselines.**
- **SHAP top features:** `gap`, `ret_1` (momentum), `close_vs_sma50` (trend), `bb_width`, `rsi_14`, `vix_change` (fear → DOWN), plus low `stoch_k` (oversold bounce → UP).

## Final message
Daily NIFTY direction is very close to random once data leakage is removed. Our models learned real but weak effects. The honest result (~52-54%) is more useful than the inflated 70-90% claims seen online, which almost always come from leakage.

---

## Likely viva questions

**Q: Why not predict the price directly?**
Price prediction is mostly "tomorrow ≈ today", which looks accurate but is useless. What a trader needs is the direction.

**Q: Why not shuffle the data in train/test split?**
Shuffling puts future days in training. In real life you only have the past.

**Q: How did you avoid data leakage?**
1. Every feature uses data up to day t only.
2. Foreign markets are lagged one day.
3. The split is by time.
4. TimeSeriesSplit is used for CV.
5. Scalers live inside the pipeline (fitted on training folds only).
6. Correlated features were dropped using training data only.
7. The final model was chosen by validation score, not test score.

**Q: Your accuracy is only ~52%. Is the project a failure?**
No. The market is close to efficient: public indicators are used by everyone, so easy patterns disappear. EDA showed near-zero autocorrelation before any model was built. A 52-54% result with honest validation is realistic. Claims of 80-90% usually mean leakage.

**Q: Why ROC-AUC and not just accuracy?**
Classes are imbalanced (53% UP), so "always UP" already gets ~53% accuracy. AUC checks whether the model ranks UP days above DOWN days, independent of the threshold.

**Q: Difference between L1 and L2?**
L1 = sum of |w|; it pushes weights to exactly 0 (feature selection). L2 = sum of w²; it shrinks all weights smoothly. L2 was better here because the signal is spread over many features.

**Q: Bagging vs boosting?**
Bagging trains trees in parallel on bootstrap samples and averages them, which reduces variance. Boosting trains trees one after another, each fixing the previous errors, which reduces bias. On noisy data, variance is the bigger problem, so bagging (RF) won.

**Q: Voting vs stacking?**
Soft voting averages the probabilities equally. Stacking trains a meta-model on out-of-fold predictions to learn how much to trust each model.

**Q: Why did SVM have the best test AUC but you didn't choose it?**
Model selection must use validation data only. Picking by test score makes the test set a second validation set, and the reported score becomes optimistic. SVM also overfit (86% train vs 48% test) and predicted UP on almost every day.

**Q: What is walk-forward validation?**
Retrain every 63 trading days (~3 months) on all past data, then predict the next 63 days. This copies how the model would be used live.

**Q: What is SHAP?**
A method from game theory (Shapley values) that splits each prediction into contributions from each feature. It shows which features pushed the probability up or down, and by how much.

**Q: What would you improve?**
1. Add non-price data (FII/DII flows, options put-call ratio, news sentiment).
2. Predict volatility, which is clearly predictable.
3. Use longer horizons (weekly).
4. Build a pre-open model using GIFT Nifty.
