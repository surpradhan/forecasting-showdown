"""Evaluate every model against energy.csv and log results to MLflow.

On macOS ARM, PyTorch, XGBoost/LightGBM, and Prophet/cmdstanpy each bundle
their own libomp.dylib. Loading more than one group in the same process causes
a segfault. This script therefore runs each group as an isolated subprocess.

Usage:
    uv run python scripts/run_all.py                   # all groups
    uv run python scripts/run_all.py --group tabular   # one group
    uv run python scripts/run_all.py --model lgbm      # one model
"""

import argparse
import math
import subprocess
import sys
import traceback

import mlflow

from src.config import load_config
from src.data.features import build_features
from src.data.loader import load_raw
from src.data.splits import chronological_split
from src.utils import empty_x

TARGET = "demand"
DATA_PATH = "data/energy.csv"
EXPERIMENT = "forecasting-showdown"

# ── model registry ──────────────────────────────────────────────────────────
GROUPS = ["univariate", "tabular", "sequential"]

MODEL_GROUPS: dict[str, str] = {
    "naive":         "univariate",
    "arima":         "univariate",
    "prophet":       "univariate",
    "linear":        "tabular",
    "random_forest": "tabular",
    "xgboost":       "tabular",
    "lgbm":          "tabular",
    "ensemble":      "tabular",
    "lstm":          "sequential",
    "gru":           "sequential",
    "transformer":   "sequential",
}
ALL_MODELS = list(MODEL_GROUPS.keys())


def _load_data():
    df = load_raw(DATA_PATH)
    df = build_features(df, freq="h")
    train, _val, test = chronological_split(df)
    feature_cols = [c for c in df.columns if c != TARGET]
    return (
        train[feature_cols], train[TARGET],
        test[feature_cols],  test[TARGET],
    )


def _seq_preprocess(seq_len: int):
    from src.data.windowing import window_sequences
    def fn(Xtr, ytr, Xte, yte):
        return (*window_sequences(Xtr, ytr, seq_len), *window_sequences(Xte, yte, seq_len))
    return fn


def _run(name: str, model, *args,
         experiment_name: str = EXPERIMENT, **kwargs) -> dict | None:
    from src.evaluation.runner import evaluate_model
    try:
        result = evaluate_model(model, *args, run_name=name,
                                experiment_name=experiment_name, **kwargs)
        print(f"[{name}] MAE={result['mae']:.4f}  train={result['train_time_s']:.1f}s",
              flush=True)
        return result
    except Exception:
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        print(f"[{name}] FAILED:", flush=True)
        traceback.print_exc()
        return None


def _print_table(results: list[dict]) -> None:
    if not results:
        print("\n=== Results (sorted by MAE) ===\n(no completed runs)")
        return
    cols = ["model", "mae", "rmse", "mape", "smape", "latency_s", "train_time_s"]
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in results)) for c in cols]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep    = "  ".join("-" * w for w in widths)
    print("\n=== Results (sorted by MAE) ===")
    print(header)
    print(sep)
    for r in results:
        row_vals = []
        for c, w in zip(cols, widths):
            val = r.get(c, "-")
            cell = f"{val:.4f}" if isinstance(val, float) else str(val)
            row_vals.append(cell.ljust(w))
        print("  ".join(row_vals))


# ── group runners (each imported only inside its subprocess) ────────────────

def run_univariate(only: str | None = None):
    from src.models.arima import ARIMAForecaster
    from src.models.naive import NaiveForecaster
    from src.models.prophet import ProphetForecaster

    X_train, y_train, X_test, y_test = _load_data()
    for name, cls in [
        ("naive",   NaiveForecaster),
        ("arima",   ARIMAForecaster),
        ("prophet", ProphetForecaster),
    ]:
        if only and name != only:
            continue
        print(f"[{name}] fitting...", flush=True)
        _run(name, cls(load_config(name)),
             empty_x(X_train.index), y_train,
             empty_x(X_test.index),  y_test)


def run_tabular(only: str | None = None):
    from src.models.ensemble import EnsembleForecaster
    from src.models.lgbm_model import LGBMForecaster
    from src.models.linear import LinearForecaster
    from src.models.random_forest import RandomForestForecaster
    from src.models.xgboost_model import XGBoostForecaster

    X_train, y_train, X_test, y_test = _load_data()
    for name, cls in [
        ("linear",        LinearForecaster),
        ("random_forest", RandomForestForecaster),
        ("xgboost",       XGBoostForecaster),
        ("lgbm",          LGBMForecaster),
        ("ensemble",      EnsembleForecaster),
    ]:
        if only and name != only:
            continue
        print(f"[{name}] fitting...", flush=True)
        _run(name, cls(load_config(name)), X_train, y_train, X_test, y_test)


def run_sequential(only: str | None = None):
    from src.models.gru import GRUForecaster
    from src.models.lstm import LSTMForecaster
    from src.models.transformer import TransformerForecaster

    X_train, y_train, X_test, y_test = _load_data()
    for name, cls in [
        ("lstm",        LSTMForecaster),
        ("gru",         GRUForecaster),
        ("transformer", TransformerForecaster),
    ]:
        if only and name != only:
            continue
        print(f"[{name}] fitting...", flush=True)
        cfg = load_config(name)
        _run(name, cls(cfg), X_train, y_train, X_test, y_test,
             preprocess_fn=_seq_preprocess(cfg["seq_len"]))


# ── orchestrator ────────────────────────────────────────────────────────────

def _run_group_subprocess(group: str, model: str | None = None) -> int:
    """Spawn an isolated child process for one model group (or single model)."""
    label = model or group
    print(f"\n{'='*50}", flush=True)
    print(f"  Starting: {label}", flush=True)
    print(f"{'='*50}", flush=True)
    cmd = [sys.executable, __file__, "--group", group]
    if model:
        cmd += ["--model", model]
    return subprocess.run(cmd, check=False).returncode


def _read_mlflow_results() -> list[dict]:
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return []
    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
    )
    seen: set[str] = set()
    rows = []
    for run in all_runs:
        model_name = run.data.tags.get("model", run.info.run_name)
        if model_name not in seen:
            seen.add(model_name)
            mae = run.data.metrics.get("mae")
            # Skip runs with no metrics (killed/failed mid-training)
            if mae is None:
                continue
            row = {"model": model_name}
            for k in ["mae", "rmse", "mape", "smape", "latency_s", "train_time_s"]:
                row[k] = run.data.metrics.get(k)
            rows.append(row)
    return rows


def main(group: str | None = None, model: str | None = None) -> None:
    if group is not None:
        # Group subprocess mode — run one group, optionally filtered to one model
        if group == "univariate":
            run_univariate(only=model)
        elif group == "tabular":
            run_tabular(only=model)
        elif group == "sequential":
            run_sequential(only=model)
    else:
        # Orchestrator mode — spawn isolated subprocesses for each target
        X_train, y_train, X_test, y_test = _load_data()
        print(
            f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows"
            f" | Features: {X_train.shape[1]}"
        )

        targets: list[tuple[str, str | None]]  # (group, model|None)
        if model:
            targets = [(MODEL_GROUPS[model], model)]
        else:
            targets = [(g, None) for g in GROUPS]

        failed = []
        for g, m in targets:
            rc = _run_group_subprocess(g, model=m)
            if rc != 0:
                label = m or g
                print(f"  !! '{label}' exited with code {rc}", flush=True)
                failed.append(label)

        results = _read_mlflow_results()
        if results:
            results.sort(key=lambda r: r["mae"] if r["mae"] is not None else float("inf"))
            _print_table(results)

        if failed:
            print(f"\nFailed: {failed}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=GROUPS, default=None,
                        help="Run one model group in isolation")
    parser.add_argument("--model", choices=ALL_MODELS, default=None,
                        help="Run a single model (spawns its group subprocess)")
    args = parser.parse_args()
    main(group=args.group, model=args.model)
