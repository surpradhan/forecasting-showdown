"""Tests for src/visuals/charts.py.

Runs in the default pytest suite — no PyTorch or Prophet imports, so no
libomp conflict. Uses matplotlib's non-interactive Agg backend.
"""

import matplotlib
matplotlib.use("Agg")   # must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visuals.charts import (
    forecast_overlay,
    mae_bar,
    metrics_grid,
    train_time_scatter,
)
from src.visuals import mae_bar as mae_bar_init  # verify __init__ re-export


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def results_df():
    """Minimal results DataFrame matching the real benchmark shape."""
    return pd.DataFrame({
        "model":        ["ensemble", "lgbm", "xgboost", "random_forest", "linear",
                         "gru", "lstm", "prophet", "transformer", "naive", "arima"],
        "mae":          [0.316, 0.317, 0.317, 0.320, 0.354,
                         0.437, 0.445, 0.489, 0.591, 0.779, 1.172],
        "rmse":         [0.460, 0.461, 0.461, 0.466, 0.504,
                         0.599, 0.589, 0.641, 0.750, 1.029, 1.328],
        "mape":         [43.9, 43.7, 43.9, 44.7, 49.5,
                         69.3, 76.2, 79.9, 107.5, 148.4, 245.9],
        "smape":        [34.8, 34.8, 35.1, 35.0, 39.2,
                         47.0, 49.0, 61.9, 58.5, 65.5, 89.3],
        "train_time_s": [4.0, 0.9, 0.4, 2.7, 0.004,
                         1788., 170., 2.9, 423., 0.001, 73.],
    })


@pytest.fixture()
def overlay_data():
    """Synthetic actuals + two prediction series with DatetimeIndex."""
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    rng = np.random.default_rng(42)
    y = pd.Series(np.sin(np.arange(200) * 2 * np.pi / 24) + 2.0, index=idx)
    preds = {
        "model_a": y + rng.normal(0, 0.1, 200),
        "model_b": y + rng.normal(0, 0.5, 200),
    }
    return y, preds


# ---------------------------------------------------------------------------
# __init__ re-export
# ---------------------------------------------------------------------------

def test_init_reexport():
    """src.visuals.__init__ must re-export mae_bar correctly."""
    from src.visuals import mae_bar as from_init
    from src.visuals.charts import mae_bar as from_charts
    assert from_init is from_charts


# ---------------------------------------------------------------------------
# mae_bar
# ---------------------------------------------------------------------------

class TestMaeBar:
    def test_returns_figure(self, results_df):
        fig = mae_bar(results_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_one_axes(self, results_df):
        fig = mae_bar(results_df)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_bar_count_matches_models(self, results_df):
        fig = mae_bar(results_df)
        ax = fig.axes[0]
        assert len(ax.patches) == len(results_df)
        plt.close(fig)

    def test_unknown_model_gets_fallback_color(self):
        """Models not in _MODEL_FAMILY should render without raising."""
        df = pd.DataFrame({"model": ["unknown_model"], "mae": [0.5]})
        fig = mae_bar(df)
        plt.close(fig)

    def test_single_model(self):
        df = pd.DataFrame({"model": ["lgbm"], "mae": [0.317]})
        fig = mae_bar(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# metrics_grid
# ---------------------------------------------------------------------------

class TestMetricsGrid:
    def test_returns_figure(self, results_df):
        fig = metrics_grid(results_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_default_four_axes(self, results_df):
        fig = metrics_grid(results_df)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_custom_metrics_subset(self, results_df):
        fig = metrics_grid(results_df, metrics=["mae", "rmse"])
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_bar_count_per_panel(self, results_df):
        fig = metrics_grid(results_df)
        for ax in fig.axes:
            assert len(ax.patches) == len(results_df)
        plt.close(fig)


# ---------------------------------------------------------------------------
# train_time_scatter
# ---------------------------------------------------------------------------

class TestTrainTimeScatter:
    def test_returns_figure(self, results_df):
        fig = train_time_scatter(results_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_x_axis_is_log_scale(self, results_df):
        fig = train_time_scatter(results_df)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        plt.close(fig)

    def test_point_count_matches_models(self, results_df):
        fig = train_time_scatter(results_df)
        ax = fig.axes[0]
        # scatter() adds a PathCollection; count its offsets
        collections = [c for c in ax.collections]
        total_points = sum(len(c.get_offsets()) for c in collections)
        assert total_points == len(results_df)
        plt.close(fig)

    def test_zero_train_time_handled(self):
        """train_time_s=0 must not crash on log scale (guarded to 1e-3)."""
        df = pd.DataFrame({
            "model": ["fast_model"],
            "mae": [0.3],
            "train_time_s": [0.0],
        })
        fig = train_time_scatter(df)
        plt.close(fig)


# ---------------------------------------------------------------------------
# forecast_overlay
# ---------------------------------------------------------------------------

class TestForecastOverlay:
    def test_returns_figure(self, overlay_data):
        y, preds = overlay_data
        fig = forecast_overlay(y, preds)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_line_count_is_predictions_plus_actual(self, overlay_data):
        y, preds = overlay_data
        fig = forecast_overlay(y, preds)
        ax = fig.axes[0]
        # one line per prediction + one for actual
        assert len(ax.lines) == len(preds) + 1
        plt.close(fig)

    def test_n_hours_limits_x_range(self, overlay_data):
        y, preds = overlay_data
        fig_full = forecast_overlay(y, preds, n_hours=168)
        fig_short = forecast_overlay(y, preds, n_hours=48)
        # Shorter window → narrower x range
        ax_full = fig_full.axes[0]
        ax_short = fig_short.axes[0]
        full_range = ax_full.get_xlim()[1] - ax_full.get_xlim()[0]
        short_range = ax_short.get_xlim()[1] - ax_short.get_xlim()[0]
        assert short_range < full_range
        plt.close(fig_full)
        plt.close(fig_short)

    def test_empty_predictions_dict(self, overlay_data):
        """Overlay with no predictions should render just the actual line."""
        y, _ = overlay_data
        fig = forecast_overlay(y, {})
        ax = fig.axes[0]
        assert len(ax.lines) == 1   # only the actual line
        plt.close(fig)

    def test_custom_title(self, overlay_data):
        y, preds = overlay_data
        fig = forecast_overlay(y, preds, title="My Custom Title")
        ax = fig.axes[0]
        assert ax.get_title() == "My Custom Title"
        plt.close(fig)
