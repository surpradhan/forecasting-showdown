import pytest

from src.config import load_config


def test_base_keys_present():
    cfg = load_config("naive")
    assert "horizon" in cfg
    assert "target_col" in cfg
    assert "date_col" in cfg


def test_model_key_overrides_base():
    cfg = load_config("naive")
    assert cfg["model"] == "naive"


def test_all_model_configs_load():
    models = [
        "naive", "arima", "prophet", "linear", "random_forest",
        "xgboost", "lgbm", "lstm", "gru", "transformer", "ensemble",
    ]
    for name in models:
        cfg = load_config(name)
        assert cfg["model"] == name, f"{name}.yaml missing 'model' key"


def test_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent")
