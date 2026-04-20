import time
from collections.abc import Callable
from typing import Any

import mlflow
import pandas as pd

from src.evaluation.metrics import all_metrics
from src.models.base import ForecasterBase


def evaluate_model(
    model: ForecasterBase,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: str,
    experiment_name: str = "forecasting-showdown",
    preprocess_fn: Callable[[Any, pd.Series, Any, pd.Series], tuple] | None = None,
) -> dict:
    """Fit, time, score, and log a forecaster to MLflow.

    Args:
        preprocess_fn: Optional callable applied to (X_train, y_train, X_test,
            y_test) before fit/predict. Use this to window flat feature matrices
            into 3-D sequence arrays for LSTM/GRU/Transformer models. Must return
            a 4-tuple in the same order. Timing starts after preprocessing.
    """
    mlflow.set_experiment(experiment_name)

    if preprocess_fn is not None:
        X_train, y_train, X_test, y_test = preprocess_fn(
            X_train, y_train, X_test, y_test
        )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(model.get_params())

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        latency = time.perf_counter() - t1

        # Re-attach timestamps when sequence models return a positional index
        if isinstance(y_test.index, pd.DatetimeIndex) and not isinstance(
            y_pred.index, pd.DatetimeIndex
        ):
            y_pred = y_pred.set_axis(y_test.index)

        scores = all_metrics(y_test, y_pred)
        scores["latency_s"] = round(latency, 4)
        scores["train_time_s"] = round(train_time, 4)

        mlflow.log_metrics(scores)
        mlflow.set_tag("model", run_name)

    return {"model": run_name, **scores}
