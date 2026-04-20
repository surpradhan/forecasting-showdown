# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                      # install all dependencies
uv sync --group dev                          # include dev tools (pytest, jupyter)
uv run pytest                                # tabular suite (excludes PyTorch + Prophet)
uv run pytest tests/test_base_seq.py tests/test_models_week3.py --override-ini="addopts="  # deep/PyTorch tests (run separately)
uv run pytest tests/test_models_week4.py --override-ini="addopts="    # Prophet tests (run separately)
uv run pytest tests/test_run_all.py --override-ini="addopts="         # run_all.py tests (run separately)
uv run pytest tests/test_metrics.py          # run a single test file
uv run jupyter notebook                      # launch notebooks
uv run python -m mlflow ui                   # view experiment results locally
```

> **Note — four isolated test suites:** PyTorch/XGBoost and Prophet/cmdstanpy both bundle `libomp` and
> hang or segfault when mixed with XGBoost/LightGBM in the same process on macOS ARM.
> `pyproject.toml` ignores `test_base_seq.py`, `test_models_week3.py`, `test_models_week4.py`,
> and `test_run_all.py` by default. The `pytestmark = pytest.mark.deep` tag on Week 3 tests
> is documentation only — the ignore entry in `pyproject.toml` is what actually prevents import.

## Framework decisions

| Concern | Choice |
|---|---|
| Package manager | uv + pyproject.toml, Python 3.11 |
| Data processing | pandas (native format for all forecasting libs) |
| Classical models | statsmodels (ARIMA/SARIMA), Prophet |
| ML models | XGBoost, LightGBM, scikit-learn |
| Deep learning | PyTorch (LSTM, GRU, Transformer) |
| Experiment tracking | MLflow local (`mlruns/`) |
| Config format | YAML per model in `configs/`; `_base.yaml` holds shared defaults merged via `load_config()` |
| Tabular model interface | `ForecasterBase` ABC in `src/models/base.py` — `fit(X: DataFrame, y: Series)` / `predict(X: DataFrame) → Series` |
| Sequence model interface | `SeqForecasterBase` ABC in `src/models/base_seq.py` — `fit(X_seq: ndarray[N,L,F], y: Series)` / `predict(X_seq) → Series` |
| Evaluation entry point | `evaluate_model()` in `src/evaluation/runner.py` — optional `preprocess_fn` for windowing before fit/predict |

## Key APIs

**Loading a model config** (`src/config.py`):
```python
from src.config import load_config
cfg = load_config("xgboost")  # merges _base.yaml + xgboost.yaml
```

**Data pipeline** (`src/data/`):
```python
from src.data.loader import load_raw
from src.data.features import build_features
from src.data.splits import chronological_split

df = load_raw("data/energy.csv")                  # DatetimeIndex, demand column
df = build_features(df, freq="h")                 # adds lag/rolling/calendar cols, drops NaN
train, val, test = chronological_split(df)        # 80/10/10 by default

target = "demand"
feature_cols = [c for c in df.columns if c != target]
X_train, y_train = train[feature_cols], train[target]
X_test,  y_test  = test[feature_cols],  test[target]
```

**Sequence windowing for deep models** (`src/data/windowing.py`):
```python
from src.data.windowing import window_sequences

X_seq, y_seq = window_sequences(X_train, y_train, seq_len=168, horizon=1)
# X_seq: ndarray (N-seq_len, seq_len, n_features)
# y_seq: Series  (N-seq_len,) with DatetimeIndex aligned to prediction timestamps
```

**Running evaluation — tabular** (`src/evaluation/runner.py`):
```python
from src.evaluation.runner import evaluate_model
result = evaluate_model(model, X_train, y_train, X_test, y_test, run_name="xgboost")
# returns dict: mae, rmse, mape, smape, latency_s, train_time_s
```

**Running evaluation — sequence models** (preprocess_fn wires in windowing):
```python
from src.data.windowing import window_sequences

seq_len = cfg["seq_len"]
preprocess_fn = lambda Xtr, ytr, Xte, yte: (
    *window_sequences(Xtr, ytr, seq_len),
    *window_sequences(Xte, yte, seq_len),
)
result = evaluate_model(model, X_train, y_train, X_test, y_test,
                        run_name="lstm", preprocess_fn=preprocess_fn)
```

**Metric edge cases**: `mape()` and `smape()` return `float("nan")` with a `RuntimeWarning` when all true values (or denominators) are zero — do not silently ignore NaN scores in the comparison table.

**Univariate models** (ARIMA, Prophet, Naive): pass an empty or index-only DataFrame for `X_train`/`X_test`; they use only `y_train` internally.

## Repository structure

```
data/
  energy.csv          UCI Household Power Consumption, resampled to hourly (34,168 rows, 2006–2010)
notebooks/            EDA and exploratory modeling
configs/
  _base.yaml          shared defaults (horizon, eval_window, target_col, date_col)
  naive.yaml, arima.yaml, linear.yaml, random_forest.yaml,
  xgboost.yaml, lgbm.yaml, lstm.yaml, gru.yaml, transformer.yaml,
  prophet.yaml, ensemble.yaml
src/
  config.py           load_config() — merges _base.yaml + model YAML
  data/
    loader.py         load_raw(path) → DatetimeIndex DataFrame
    features.py       build_features(df, freq, target_col) → enriched DataFrame
    splits.py         chronological_split(df, val_frac, test_frac) → train/val/test
    windowing.py      window_sequences(X, y, seq_len, horizon) → (ndarray, Series)
  models/
    base.py           ForecasterBase ABC — tabular fit(DataFrame)/predict(DataFrame)
    base_seq.py       SeqForecasterBase ABC — sequence fit(ndarray)/predict(ndarray)
                      helpers: _to_tensors, _make_loader, _run_epochs, _default_device
    naive.py          NaiveForecaster — seasonal (phase-aligned) and last-value
    arima.py          ARIMAForecaster — SARIMA via statsmodels SARIMAX
    linear.py         LinearForecaster — Ridge regression (sklearn)
    random_forest.py  RandomForestForecaster — sklearn RandomForestRegressor
    xgboost_model.py  XGBoostForecaster — XGBoost with optional early stopping
    lgbm_model.py     LGBMForecaster — LightGBM with optional early stopping
    lstm.py           LSTMForecaster — nn.LSTM + linear head
    gru.py            GRUForecaster — nn.GRU + linear head
    transformer.py    TransformerForecaster — input_proj + sinusoidal pos enc + TransformerEncoder + linear head
    ensemble.py       EnsembleForecaster — mean/weighted average over tabular sub-models
  evaluation/
    metrics.py        mae, rmse, mape, smape, all_metrics
    runner.py         evaluate_model(model, ..., preprocess_fn=None) → dict
  visuals/            Chart and report figure generation
    __init__.py       re-exports all four chart functions
    charts.py         mae_bar, metrics_grid, train_time_scatter, forecast_overlay
results/              Saved predictions and scores
reports/              Final write-up and exported figures
tests/
  conftest.py         (does not exist — see test suite note above)
  test_config.py
  test_metrics.py
  test_runner.py
  test_data_pipeline.py
  test_windowing.py
  test_models_week1.py
  test_models_week2.py
  test_base_seq.py    [deep] SeqForecasterBase + helpers — run separately
  test_models_week3.py [deep] LSTM, GRU, Transformer + runner integration — run separately
  test_integration.py # end-to-end: real data → features → split → naive → evaluate_model
scripts/              (Week 4) run_all.py — evaluates every model against energy.csv
```

## Models being benchmarked

| Status | Model |
|---|---|
| ✅ Implemented | Naive/Seasonal Naive, ARIMA/SARIMA |
| ✅ Implemented | Linear/Ridge, Random Forest, XGBoost, LightGBM |
| ✅ Implemented | LSTM, GRU, Transformer |
| ✅ Week 4 | Prophet, evaluation runs for all models, comparison notebook |
| ✅ Week 5 | EnsembleForecaster, visuals (src/visuals/charts.py), final write-up (reports/report.md) |

Each model must be evaluated on the same test window using the same preprocessing pipeline.

## Roadmap

### Week 4
1. **ProphetForecaster** — `src/models/prophet.py` + `configs/prophet.yaml` + tests in `test_models_week4.py`
2. **Evaluation script** — `scripts/run_all.py` runs every model (Naive → Prophet) through
   `evaluate_model()` against the same chronological train/test split, logs all runs to MLflow
3. **Comparison notebook** — `notebooks/results.ipynb` reads MLflow runs, renders
   MAE/RMSE/MAPE/SMAPE/latency table sorted by MAE

### Week 5
1. **EnsembleForecaster** ✅ — mean/weighted average over configurable tabular members; `src/models/ensemble.py` + `configs/ensemble.yaml`; benchmarked MAE=0.3157 (best overall)
2. **Visuals** ✅ — `src/visuals/charts.py`: `mae_bar`, `metrics_grid`, `train_time_scatter`, `forecast_overlay`; figures saved to `reports/figures/`; wired into `notebooks/results.ipynb`
3. **Final write-up** ✅ — `reports/report.md`: full comparison table, per-model interpretability notes, key findings, recommendations

## Scripts (Week 4+)

```
scripts/
  run_all.py    Runs every model end-to-end and logs results to MLflow.
                Usage: uv run python scripts/run_all.py                    # all groups
                       uv run python scripts/run_all.py --group tabular    # one group
                       uv run python scripts/run_all.py --model lgbm       # one model
                Groups: univariate (naive/arima/prophet), tabular (linear/rf/xgb/lgbm),
                        sequential (lstm/gru/transformer). Each group runs in its own
                        subprocess to avoid libomp conflicts on macOS ARM.
notebooks/
  results.ipynb Reads MLflow runs, renders comparison table and forecast plots.
```

## Evaluation standard

Every model is scored on: MAE, RMSE, MAPE, SMAPE, forecast latency, and training time. Interpretability notes are recorded per model. All comparisons use a chronological train/val/test split (no random shuffling of time series data).

## Feature engineering conventions

- `build_features(df, freq="h", target_col="demand")` handles all feature creation
- Lags are frequency-aware: hourly → [1, 24, 168]; daily → [1, 7, 30]; 15-min → [4, 96, 672]
- Rolling window (mean, std, min, max) uses `shift(1)` before rolling to prevent data leakage
- Calendar features: hour, day_of_week, month, is_weekend, season (0=winter…3=autumn)
- `dropna()` is called inside `build_features` — the first 168 rows (hourly) are consumed as warm-up

## Sequence model conventions (Week 3)

- All deep models subclass `SeqForecasterBase` from `src/models/base_seq.py`
- Use `window_sequences(X, y, seq_len)` via `preprocess_fn` in `evaluate_model()` — do not window inside the model
- `seq_len`, `hidden_size` / `d_model` / `nhead`, `epochs`, `batch_size`, `lr` all come from config
- `_default_device()` returns MPS on Apple Silicon, CUDA if available, else CPU
- Tests for Week 3 models go in `test_models_week3.py` with `pytestmark = pytest.mark.deep`
- **MPS compilation note**: on Apple Silicon, PyTorch compiles Metal shaders on first use per
  model type. This one-time cost (~2–7 min) dominates training time regardless of epoch count.
  Configs use `epochs: 20` as a reasonable default; re-runs are faster once shaders are cached.

## Forecasting setup

- Start with one-step-ahead prediction; extend to multi-step only after baselines are stable
- Use rolling-origin evaluation for robust benchmarking
- Keep target variable and evaluation window identical across all models
