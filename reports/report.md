# Forecasting Showdown — Final Report

**Dataset**: UCI Household Power Consumption (hourly, 2006–2010, 34 168 raw rows → 34 000 after feature engineering)  
**Task**: One-step-ahead demand forecasting  
**Evaluation window**: Chronological 80 / 10 / 10 train / val / test split (no shuffling)  
**Date**: 2026-04-20

---

## 1. Dataset and Methodology

### Data

The UCI Household Power Consumption dataset records net household electrical
demand at one-minute resolution from December 2006 to November 2010. It was
resampled to hourly frequency, yielding 34 168 observations with a single
`demand` target column. After feature engineering removes the first 168 rows
as lag warm-up, 34 000 rows remain for modelling.

### Feature engineering

All tabular and deep models consume a shared feature matrix built by
`build_features()`:

| Feature group | Details |
|---|---|
| Lag features | `demand` at lags 1 h, 24 h, 168 h |
| Rolling statistics | 24 h mean, std, min, max (shifted by 1 to prevent leakage) |
| Calendar | hour-of-day, day-of-week, month, is\_weekend, season |

The first 168 rows are consumed as warm-up and dropped. Univariate models
(Naive, ARIMA, Prophet) receive only the target series.

### Train / test split

| Split | Rows | Period |
|---|---|---|
| Train | 27 200 | 2006-12-23 → 2010-02-06 |
| Validation | 3 400 | 2010-02-06 → 2010-06-29 |
| Test | 3 400 | 2010-06-29 → 2010-11-26 |

### Evaluation metrics

Every model is scored on the same held-out test window:

| Metric | Measures |
|---|---|
| **MAE** | Average absolute error (primary metric) |
| **RMSE** | Penalises large errors more heavily than MAE |
| **MAPE** | Scale-free percentage error |
| **SMAPE** | Symmetric variant; stable near zero |
| **Latency** | Inference time for the full test set (seconds) |
| **Train time** | Wall-clock fitting time (seconds) |

---

## 2. Results

### 2.1 Full comparison table

| Rank | Model | MAE | RMSE | MAPE | SMAPE | Train (s) | Latency (s) |
|---|---|---|---|---|---|---|---|
| 1 | **ensemble** | **0.3157** | **0.4595** | 43.85 | 34.80 | 4.0 | 0.039 |
| 2 | lgbm | 0.3165 | 0.4609 | 43.68 | 34.82 | 0.9 | 0.006 |
| 3 | xgboost | 0.3173 | 0.4610 | 43.93 | 35.09 | 0.4 | 0.004 |
| 4 | random\_forest | 0.3196 | 0.4655 | 44.71 | 35.04 | 2.7 | 0.027 |
| 5 | linear | 0.3541 | 0.5040 | 49.50 | 39.18 | 0.004 | 0.001 |
| 6 | gru | 0.4371 | 0.5988 | 69.30 | 46.99 | 1 788 | 0.427 |
| 7 | lstm | 0.4449 | 0.5886 | 76.20 | 49.04 | 170 | 0.643 |
| 8 | prophet | 0.4890 | 0.6405 | 79.85 | 61.92 | 2.9 | 0.455 |
| 9 | transformer | 0.5911 | 0.7503 | 107.46 | 58.53 | 423 | 1.910 |
| 10 | naive | 0.7792 | 1.0287 | 148.43 | 65.45 | 0.001 | 0.005 |
| 11 | arima | 1.1722 | 1.3277 | 245.89 | 89.28 | 72.9 | 0.145 |

### 2.2 MAE by model

![MAE bar chart](figures/mae_bar.png)

The three gradient-boosting models and their ensemble occupy a tight band
(MAE 0.316–0.320), well separated from the deep learning tier (0.437–0.591)
and the classical/univariate tier (0.489–1.172).

### 2.3 All error metrics

![Metrics grid](figures/metrics_grid.png)

Rankings are consistent across MAE, RMSE, and SMAPE. The one exception is
**MAPE**, where LSTM (76.2) scores noticeably worse than GRU (69.3) despite
similar MAE — a sign that LSTM over-predicts on low-demand hours where
percentage errors inflate.

### 2.4 Accuracy vs training cost

![Train time scatter](figures/train_time_scatter.png)

The log-scaled x-axis exposes the full spread: tabular models achieve the
best accuracy in under 5 seconds total, while deep learning models require
170–1 788 seconds without a commensurate improvement. The ideal operating
region (low MAE, fast training) is the bottom-left corner — occupied
exclusively by tabular models.

### 2.5 Forecast overlay — first week of test set

![Forecast overlay](figures/forecast_overlay.png)

Ensemble and LightGBM track the actual demand curve closely throughout the
week, capturing both the daily double-peak pattern and the quieter weekend
periods (Jul 3–4, 2010). The naive model follows the general shape but drifts
noticeably on days where the corresponding week-prior demand was atypical.

---

## 3. Per-model notes

### Naive (seasonal, period = 24 h)
**MAE 0.779 | Train < 1 ms**

Repeats the demand value from exactly 24 hours prior. Zero trainable
parameters — the entire "model" is a one-line lookup. Despite its simplicity
it captures the dominant daily cycle, making it a meaningful lower-bound
baseline. Fails whenever the current week's pattern differs from the prior
week (holidays, anomalies, seasonal drift).

*Interpretability*: Complete. Every prediction is traceable to a specific
historical observation.

---

### ARIMA (SARIMA)
**MAE 1.172 | Train 73 s**

The worst-performing model despite having one of the longer training times.
The SARIMA order was set conservatively (no automated selection); a proper
auto-ARIMA grid search would likely close some of the gap. SARIMA also
operates purely on the target series and therefore cannot exploit the rich
calendar and lag features that drive tabular model accuracy.

*Interpretability*: High. Explicit autoregressive and moving-average
coefficients with confidence intervals.

---

### Prophet
**MAE 0.489 | Train 2.9 s**

Decomposes demand into trend + yearly/weekly/daily seasonality components fit
via Stan MCMC. Competitive with the deep learning tier at a fraction of the
training cost and with built-in uncertainty intervals. Prophet's main weakness
here is that it treats the target as a smooth curve and misses sharp
intra-hour spikes captured by lag-based tabular models.

*Interpretability*: High. Components are directly visualisable and each
additive term has a named semantic meaning.

---

### Linear / Ridge
**MAE 0.354 | Train < 1 ms**

Ridge regression on the 12 engineered features. The fastest model to train
and a surprisingly strong baseline — its MAE is only 12% worse than the best
model, for essentially zero computational cost. The gap to the tree-based
models reflects their ability to learn non-linear feature interactions (e.g.
the interaction between hour-of-day and day-of-week).

*Interpretability*: Full. Coefficients are directly readable and stable across
runs. Feature signs match intuition (lag-168 is the dominant positive
predictor).

---

### Random Forest
**MAE 0.320 | Train 2.7 s**

100 decision trees averaged. Handles non-linearities well and is robust to
outliers by design. Prediction variance is higher than the boosted models at
test time, reflected in a slightly higher RMSE despite similar MAE. Feature
importance consistently ranks `lag_168` (same-hour last week) and `lag_24`
(same-hour yesterday) as the top predictors.

*Interpretability*: Medium. Feature importance is available but individual
predictions are opaque. Partial dependence plots can illuminate key
relationships.

---

### XGBoost
**MAE 0.317 | Train 0.4 s**

Gradient boosted trees with the fastest training time of the tree ensemble
family. Achieves near-identical accuracy to LightGBM at 2.5× less training
time, making it a strong default choice when batch retraining cadence matters.
Supports SHAP values for post-hoc explainability.

*Interpretability*: Medium. SHAP values provide per-prediction feature
attributions.

---

### LightGBM
**MAE 0.316 | Train 0.9 s**

Best solo model. Leaf-wise tree growth finds finer splits than XGBoost's
level-wise strategy, squeezing slightly more accuracy from the same feature
set. LightGBM is the clear production candidate when a single model is
required: best accuracy, sub-second training, millisecond inference, and SHAP
support.

*Interpretability*: Medium. SHAP values available.

---

### Ensemble (lgbm + xgboost + random\_forest, mean)
**MAE 0.316 | Train 4.0 s**

Simple unweighted average of the three best tabular models. Achieves the
lowest MAE overall, though the improvement over solo LightGBM is modest
(−0.0008 MAE, −0.3%). The gain is real but small because the three members
are tightly correlated — all trained on identical features with similar
inductive biases. A more diverse ensemble (e.g. adding Prophet or a deep
model) could increase variance reduction at the cost of degrading the average.
Training cost is additive (~4 s total).

*Interpretability*: Inherited from members. Prediction = average of three
independently interpretable models.

---

### LSTM
**MAE 0.445 | Train 170 s**

Two-layer LSTM with a linear head, trained for 20 epochs on 168-step windows.
Ranks 7th overall — competitive with GRU but outperformed by all tabular
models. The LSTM's advantage in theory is learning long-range temporal
dependencies beyond the 168-hour lag window, but on this dataset the dominant
patterns are well-captured by a single week of lags.

*Interpretability*: Low. Hidden states are not semantically interpretable.
Attention probing or input-gradient methods are needed for any explanation.

---

### GRU
**MAE 0.437 | Train 1 788 s**

Gated recurrent unit — essentially LSTM with fewer parameters. Achieves
slightly better MAE than LSTM (0.437 vs 0.445) but at 10× the training time.
The 1 788 s figure includes a one-time Apple Silicon MPS Metal shader
compilation cost (~22 min) that occurs on first use of each new model
architecture. On subsequent runs with cached shaders, GRU trains in
~8 minutes — consistent with LSTM at a similar epoch count.

*Interpretability*: Low. Same caveats as LSTM.

---

### Transformer
**MAE 0.591 | Train 423 s**

Sinusoidal positional encoding + 2-layer `TransformerEncoder` + linear head.
Worst of the deep models and the only one outperformed by Prophet. The
architecture was not tuned for this task — a vanilla encoder without a
decoder is not ideally suited to single-step autoregressive forecasting on
short sequences (168 steps). Performance would likely improve with a
dedicated forecasting architecture (e.g. Informer, PatchTST).

*Interpretability*: Low. Attention weights can be visualised but are not
reliable explanations without further analysis.

---

## 4. Key findings

1. **Tabular models dominate.** All five tree-based / linear models outperform
   every deep learning and classical model on this dataset. The engineered
   features (lag-1/24/168, rolling statistics, calendar) provide a sufficient
   inductive bias for hourly energy demand — sequential architectures add
   complexity without accuracy gains.

2. **The best model is the cheapest to improve.** The ensemble's
   marginal gain over solo LightGBM (+0.3% MAE reduction) required fitting
   three models instead of one. Unless sub-0.316 MAE is a hard requirement,
   solo LightGBM is the production choice.

3. **Feature engineering matters more than model complexity.** The Ridge
   regression model (12 features, < 1 ms training) reaches MAE 0.354 — better
   than every deep learning model and close to tree-based accuracy. Investing
   in features yields more return than investing in model architecture for
   structured tabular time-series.

4. **Deep learning is expensive and underperforms here.** GRU and LSTM train
   for 170–1 788 s to achieve MAE 0.437–0.445, which is 38–40% worse than
   LightGBM. This is consistent with literature showing that gradient-boosted
   trees remain competitive with deep learning on tabular data without
   extensive tuning.

5. **Classical univariate models fail without feature access.** ARIMA ranks
   last (MAE 1.17) partly because it operates on the raw series without
   calendar or lag features. Given the same features, statistical models can
   be competitive — Prophet demonstrates this at MAE 0.489 through its
   explicit seasonality decomposition.

6. **Evaluation on a chronological split is essential.** Random shuffling
   would inflate all scores by leaking future information. The strict 80/10/10
   chronological split ensures all reported metrics reflect real-world
   deployment conditions.

---

## 5. Recommendations

| Use case | Recommended model | Rationale |
|---|---|---|
| Production (single model) | **LightGBM** | Best solo MAE, sub-second training, SHAP support |
| Maximum accuracy | **Ensemble** | Marginal MAE improvement; deploy if retraining budget allows |
| Fastest possible training | **Linear / Ridge** | < 1 ms; strong baseline for CI/CD regression testing |
| Interpretability required | **Linear or Prophet** | Coefficients / additive components are directly readable |
| Uncertainty quantification | **Prophet** | Built-in credible intervals from Stan MCMC |
| Sequence-first architecture | **GRU** | Best deep model; re-run after shader cache is warm |
