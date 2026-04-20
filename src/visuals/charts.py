"""Reusable chart functions for the forecasting benchmark.

Design principles applied
-------------------------
- Okabe-Ito colorblind-safe palette, families encoded consistently
- Spines removed (Tufte data-ink ratio)
- Direct value labels on bar charts — no squinting at axes
- Log-scale x-axis on the scatter (train times span 5 orders of magnitude)
- Overlay uses line-style + colour together for dual encoding

All functions return a ``matplotlib.figure.Figure`` so they can be
displayed inline in Jupyter or saved to disk (``fig.savefig(...)``).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette (doi:10.1038/nmeth.1618)
_FAMILY_COLORS: dict[str, str] = {
    "tabular":    "#0072B2",   # blue
    "deep":       "#E69F00",   # orange
    "univariate": "#009E73",   # teal
}

_FAMILY_LABELS: dict[str, str] = {
    "tabular":    "Tabular",
    "deep":       "Deep Learning",
    "univariate": "Univariate / Classical",
}

_MODEL_FAMILY: dict[str, str] = {
    "ensemble":      "tabular",
    "lgbm":          "tabular",
    "xgboost":       "tabular",
    "random_forest": "tabular",
    "linear":        "tabular",
    "lstm":          "deep",
    "gru":           "deep",
    "transformer":   "deep",
    "prophet":       "univariate",
    "naive":         "univariate",
    "arima":         "univariate",
}

# Line styles for the forecast overlay.
# Each model gets a distinct colour AND line style (dual encoding → readable in greyscale).
# Colours are from the Okabe-Ito palette; chosen to be maximally distinct.
_OVERLAY_STYLES: list[dict] = [
    {"color": "#0072B2", "linestyle": "-",  "linewidth": 1.5},   # blue      — solid
    {"color": "#D55E00", "linestyle": "--", "linewidth": 1.4},   # vermilion — dashed
    {"color": "#009E73", "linestyle": ":",  "linewidth": 1.4},   # teal      — dotted
    {"color": "#E69F00", "linestyle": "-.", "linewidth": 1.4},   # orange    — dash-dot
    {"color": "#CC79A7", "linestyle": "--", "linewidth": 1.4},   # pink      — dashed
]


def _bar_colors(models: pd.Series) -> list[str]:
    """Return a bar colour for each model based on its family."""
    return [_FAMILY_COLORS.get(_MODEL_FAMILY.get(m, ""), "#888888") for m in models]


def _family_legend(ax: plt.Axes, families: set[str]) -> None:
    """Append a compact family-colour legend to the axes."""
    handles = [
        mpatches.Patch(color=_FAMILY_COLORS[f], label=_FAMILY_LABELS[f])
        for f in ["tabular", "deep", "univariate"]
        if f in families
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.85,
              fontsize=8, title="Model family", title_fontsize=8)


def _despine(ax: plt.Axes) -> None:
    """Remove top and right spines (Tufte)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1. MAE bar chart
# ---------------------------------------------------------------------------

def mae_bar(
    df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "MAE by model",
) -> plt.Figure:
    """Horizontal bar chart of MAE, colour-coded by model family.

    Parameters
    ----------
    df:    Results DataFrame with ``model`` and ``mae`` columns, sorted ascending.
    ax:    Optional existing Axes to draw into.
    title: Chart title.
    """
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure  # type: ignore[assignment]

    plot_df = df.iloc[::-1].reset_index(drop=True)   # worst → best (top → bottom)
    models  = plot_df["model"]
    values  = plot_df["mae"]
    colors  = _bar_colors(models)

    bars = ax.barh(models, values, color=colors, height=0.65, edgecolor="white", linewidth=0.4)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8.5, color="#333333",
        )

    # Reference line at best (ensemble) value
    best = values.max()   # reversed, so max in plot_df is the worst; use min of original
    best_val = df["mae"].min()
    ax.axvline(best_val, color="#333333", linewidth=0.8, linestyle="--", alpha=0.5,
               label=f"Best: {best_val:.3f}")

    ax.set_xlabel("MAE (lower is better)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, values.max() * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.4, color="#aaaaaa")
    ax.tick_params(axis="y", labelsize=9)
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    families = {_MODEL_FAMILY.get(m, "") for m in models}
    _family_legend(ax, families)

    if owns_fig:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Metrics grid (MAE / RMSE / MAPE / SMAPE)
# ---------------------------------------------------------------------------

def metrics_grid(
    df: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
) -> plt.Figure:
    """2 × 2 grid of bar charts for the four error metrics.

    Parameters
    ----------
    df:      Results DataFrame sorted by MAE ascending.
    metrics: Metric columns (default: mae, rmse, mape, smape).
    """
    if metrics is None:
        metrics = ["mae", "rmse", "mape", "smape"]

    n = len(metrics)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes_flat = np.array(axes).flatten()

    plot_df = df.iloc[::-1].reset_index(drop=True)
    models  = plot_df["model"]
    colors  = _bar_colors(models)

    for ax, metric in zip(axes_flat, metrics):
        values = plot_df[metric]
        bars = ax.barh(models, values, color=colors, height=0.65,
                       edgecolor="white", linewidth=0.4)

        # Value labels on every panel
        for bar, val in zip(bars, values):
            ax.text(
                val + values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=7.5, color="#333333",
            )

        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.set_xlim(0, values.max() * 1.22)
        ax.grid(axis="x", linestyle=":", alpha=0.4, color="#aaaaaa")
        ax.tick_params(axis="y", labelsize=8, length=0)
        _despine(ax)
        ax.spines["left"].set_visible(False)

    # Shared family legend in first panel
    families = {_MODEL_FAMILY.get(m, "") for m in models}
    _family_legend(axes_flat[0], families)

    fig.suptitle("Error metrics by model (lower is better)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Train time vs MAE scatter
# ---------------------------------------------------------------------------

def train_time_scatter(
    df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Accuracy vs Training Cost",
) -> plt.Figure:
    """Scatter of log-scale train time (x) vs MAE (y), colour-coded by family.

    The x-axis is log-scaled because train times span five orders of magnitude
    (milliseconds → half-hour). Direct labels replace a separate legend.

    Parameters
    ----------
    df:    Results DataFrame with ``model``, ``mae``, ``train_time_s``.
    ax:    Optional existing Axes.
    title: Chart title.
    """
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure  # type: ignore[assignment]

    families = set()
    for _, row in df.iterrows():
        fam   = _MODEL_FAMILY.get(row["model"], "")
        color = _FAMILY_COLORS.get(fam, "#888888")
        families.add(fam)
        ax.scatter(
            max(row["train_time_s"], 1e-3),   # guard against 0 on log scale
            row["mae"],
            s=90, color=color, zorder=4, edgecolors="white", linewidths=0.6,
        )

    # Label offsets tuned to the actual log-scale positions to prevent overlap.
    # The four top tabular models cluster between 0.3–5 s at MAE ≈ 0.32,
    # so we alternate above/below to keep them from stacking.
    _offsets: dict[str, tuple[float, float]] = {
        "ensemble":      (  6,   6),   # above-right
        "lgbm":          (  6, -13),   # below-right
        "xgboost":       (  6,   6),   # above-right
        "random_forest": (  6, -13),   # below-right
        "linear":        (  6,   6),
        "lstm":          (  6,   6),
        "gru":           (  6, -13),
        "transformer":   (  6,   6),
        "prophet":       (  6, -13),
        "naive":         (  6,   6),
        "arima":         (  6,   6),
    }
    for _, row in df.iterrows():
        name  = row["model"]
        fam   = _MODEL_FAMILY.get(name, "")
        color = _FAMILY_COLORS.get(fam, "#333333")
        dx, dy = _offsets.get(name, (6, 4))
        ax.annotate(
            name,
            (max(row["train_time_s"], 1e-3), row["mae"]),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=8.5, color=color, fontweight="medium",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Train time — log scale (seconds)", fontsize=10)
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(which="both", linestyle=":", alpha=0.35, color="#aaaaaa")
    ax.grid(which="major", linestyle=":", alpha=0.5, color="#999999")
    _despine(ax)

    # Family legend
    handles = [
        mpatches.Patch(color=_FAMILY_COLORS[f], label=_FAMILY_LABELS[f])
        for f in ["tabular", "deep", "univariate"]
        if f in families
    ]
    ax.legend(handles=handles, framealpha=0.85, fontsize=8,
              title="Model family", title_fontsize=8)

    if owns_fig:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Forecast overlay
# ---------------------------------------------------------------------------

def forecast_overlay(
    y_test: pd.Series,
    predictions: dict[str, pd.Series],
    *,
    n_hours: int = 168,
    title: str = "Forecast vs Actuals — first week of test set",
) -> plt.Figure:
    """Overlay model predictions against actuals for a one-week test window.

    Actual demand is plotted in black with the highest z-order. Each model
    gets a distinct colour + line-style combination (dual encoding) so the
    chart remains readable in greyscale.

    Parameters
    ----------
    y_test:      Ground-truth demand Series with DatetimeIndex.
    predictions: ``{model_name: pred_series}`` with the same index as ``y_test``.
    n_hours:     Hours to show (default 168 = one week).
    title:       Chart title.
    """
    window_idx = y_test.index[:n_hours]
    actual = y_test.loc[window_idx]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Set a generous y-axis ceiling (2× the 99th percentile of actuals) so that
    # the axis is anchored to the data distribution rather than the plot default.
    # Prediction series are soft-clipped at 1.1× this ceiling to prevent extreme
    # model spikes from compressing the rest of the chart.
    y_ceil = np.percentile(actual.values, 99) * 2.0
    y_floor = max(0, actual.min() * 0.85)

    # Draw predictions first (behind actuals)
    for (name, pred), style in zip(predictions.items(), _OVERLAY_STYLES):
        ax.plot(
            window_idx,
            pred.loc[window_idx].clip(lower=y_floor, upper=y_ceil * 1.1).values,
            label=name,
            alpha=0.80,
            **style,
        )

    # Actuals on top — heavier weight and pure black so it reads above all predictions
    ax.plot(
        window_idx, actual.values,
        color="#000000", linewidth=2.6, label="actual", zorder=10,
    )

    ax.set_ylim(y_floor, y_ceil)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Demand (kW)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Date formatting
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35, color="#aaaaaa")
    _despine(ax)

    fig.tight_layout()
    return fig
