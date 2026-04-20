"""End-to-end smoke test: real data file → features → split → model → evaluate_model."""
import pytest

from src.utils import empty_x

DATA_PATH = "data/energy.csv"


@pytest.fixture(scope="module")
def pipeline():
    """Load data and return (X_train, y_train, X_test, y_test)."""
    import pandas as pd
    from src.data.loader import load_raw
    from src.data.features import build_features
    from src.data.splits import chronological_split

    df = load_raw(DATA_PATH)
    df = build_features(df, freq="h")

    train, _val, test = chronological_split(df, val_frac=0.1, test_frac=0.1)

    target = "demand"
    feature_cols = [c for c in df.columns if c != target]

    X_train = train[feature_cols]
    y_train = train[target]
    X_test  = test[feature_cols]
    y_test  = test[target]

    return X_train, y_train, X_test, y_test


def test_pipeline_shapes(pipeline):
    X_train, y_train, X_test, y_test = pipeline
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) > len(X_test)
    assert X_train.isnull().sum().sum() == 0


def test_naive_end_to_end(pipeline):
    from src.models.naive import NaiveForecaster
    from src.evaluation.runner import evaluate_model

    X_train, y_train, X_test, y_test = pipeline
    model = NaiveForecaster({"strategy": "seasonal", "seasonal_period": 24})

    # Naive uses only y_train; pass empty X
    result = evaluate_model(model, empty_x(X_train.index), y_train,
                            empty_x(X_test.index), y_test,
                            run_name="naive_integration")

    assert result["mae"] > 0
    assert result["rmse"] > 0
    assert result["train_time_s"] >= 0
    assert result["latency_s"] >= 0
