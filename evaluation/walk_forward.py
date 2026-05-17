"""Walk-forward research pipeline and production monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from evaluation.metrics import information_coefficient, rolling_information_coefficient
from evaluation.performance import max_drawdown, sharpe_ratio, signal_strategy_returns


ModelFactory = Callable[[], Any]


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for calendar-based walk-forward validation."""

    initial_train_years: int = 2
    test_months: int = 3
    validation_fraction: float = 0.2
    expanding: bool = True
    min_train_observations: int = 30
    min_test_observations: int = 5


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward train/test window."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample walk-forward predictions and diagnostics."""

    predictions: pd.DataFrame
    metrics: dict[str, float]
    rolling_ic: pd.Series
    windows: pd.DataFrame


def _validate_datetime_index(data: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    """Return a sorted DatetimeIndex or raise a helpful error."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Walk-forward validation requires a DatetimeIndex.")
    if data.index.hasnans:
        raise ValueError("DatetimeIndex contains NaT values.")
    return data.sort_index().index


def generate_walk_forward_windows(
    data: pd.DataFrame | pd.Series,
    config: WalkForwardConfig | None = None,
) -> list[WalkForwardWindow]:
    """Generate calendar-based walk-forward train/test windows.

    The first window trains on ``initial_train_years`` and tests on the next
    ``test_months``. Each subsequent window slides forward by ``test_months``.
    """
    cfg = config or WalkForwardConfig()
    if cfg.initial_train_years <= 0 or cfg.test_months <= 0:
        raise ValueError("initial_train_years and test_months must be positive.")

    index = _validate_datetime_index(data)
    sorted_index = index.sort_values()
    first_timestamp = sorted_index[0]
    last_timestamp = sorted_index[-1]
    train_window = pd.DateOffset(years=cfg.initial_train_years)
    test_window = pd.DateOffset(months=cfg.test_months)

    windows: list[WalkForwardWindow] = []
    train_anchor = first_timestamp
    train_end = first_timestamp + train_window
    window_id = 0

    while train_end < last_timestamp:
        test_start = train_end
        test_end = min(test_start + test_window, last_timestamp + pd.Timedelta(nanoseconds=1))
        train_start = first_timestamp if cfg.expanding else train_anchor

        train_mask = (index >= train_start) & (index < train_end)
        test_mask = (index >= test_start) & (index < test_end)
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)

        if (
            len(train_indices) >= cfg.min_train_observations
            and len(test_indices) >= cfg.min_test_observations
        ):
            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    train_start=pd.Timestamp(train_start),
                    train_end=pd.Timestamp(train_end),
                    test_start=pd.Timestamp(test_start),
                    test_end=pd.Timestamp(test_end),
                    train_indices=train_indices,
                    test_indices=test_indices,
                ),
            )
            window_id += 1

        if not cfg.expanding:
            train_anchor = train_anchor + test_window
        train_end = train_end + test_window

    return windows


def run_walk_forward_model(
    x: pd.DataFrame,
    y: pd.Series,
    model_factory: ModelFactory,
    config: WalkForwardConfig | None = None,
    realized_returns: pd.Series | None = None,
    signal_threshold: float = 0.0,
    transaction_cost_bps: float = 0.0,
    rolling_ic_window: int = 20,
) -> WalkForwardResult:
    """Train/predict across walk-forward windows and evaluate OOS only."""
    cfg = config or WalkForwardConfig()
    x = x.sort_index()
    y = y.reindex(x.index)
    realized = realized_returns.reindex(x.index) if realized_returns is not None else y
    windows = generate_walk_forward_windows(x, cfg)
    rows: list[pd.DataFrame] = []
    window_rows: list[dict[str, Any]] = []

    for window in windows:
        train_x_full = x.iloc[window.train_indices]
        train_y_full = y.iloc[window.train_indices]
        test_x = x.iloc[window.test_indices]
        test_y = y.iloc[window.test_indices]

        validation_count = int(len(train_x_full) * cfg.validation_fraction)
        model = model_factory()

        if validation_count >= 2 and len(train_x_full) - validation_count >= cfg.min_train_observations:
            fit_x = train_x_full.iloc[:-validation_count]
            fit_y = train_y_full.iloc[:-validation_count]
            valid_x = train_x_full.iloc[-validation_count:]
            valid_y = train_y_full.iloc[-validation_count:]
            try:
                model.fit(fit_x, fit_y, valid_x, valid_y)
            except TypeError:
                model.fit(train_x_full, train_y_full)
        else:
            model.fit(train_x_full, train_y_full)

        predictions = model.predict(test_x)
        rows.append(
            pd.DataFrame(
                {
                    "prediction": predictions.reindex(test_x.index),
                    "realized_return": realized.iloc[window.test_indices],
                    "target": test_y,
                    "window_id": window.window_id,
                },
                index=test_x.index,
            ),
        )
        window_rows.append(
            {
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "train_observations": int(len(window.train_indices)),
                "test_observations": int(len(window.test_indices)),
            },
        )

    if not rows:
        raise ValueError("No valid walk-forward windows were generated.")

    predictions_frame = pd.concat(rows).sort_index()
    strategy_returns = signal_strategy_returns(
        predictions_frame["prediction"],
        predictions_frame["realized_return"],
        threshold=signal_threshold,
        transaction_cost_bps=transaction_cost_bps,
    )
    rolling_ic = rolling_information_coefficient(
        predictions_frame["prediction"],
        predictions_frame["realized_return"],
        window=rolling_ic_window,
    )
    metrics = {
        "oos_observations": float(len(predictions_frame)),
        "oos_ic": information_coefficient(
            predictions_frame["prediction"],
            predictions_frame["realized_return"],
        ),
        "oos_sharpe": sharpe_ratio(strategy_returns),
        "oos_max_drawdown": max_drawdown(strategy_returns),
        "mean_rolling_ic": float(rolling_ic.mean()) if not rolling_ic.dropna().empty else float("nan"),
    }

    return WalkForwardResult(
        predictions=predictions_frame,
        metrics=metrics,
        rolling_ic=rolling_ic,
        windows=pd.DataFrame(window_rows),
    )


def population_stability_index(
    expected: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate Population Stability Index between train and test distributions."""
    expected_series = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual_series = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()
    if expected_series.empty or actual_series.empty:
        return float("nan")
    if expected_series.nunique() <= 1:
        expected_value = expected_series.iloc[0]
        return 0.0 if actual_series.nunique() <= 1 and actual_series.iloc[0] == expected_value else 10.0

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(expected_series.quantile(quantiles).to_numpy())
    if len(breakpoints) <= 2:
        minimum = min(expected_series.min(), actual_series.min())
        maximum = max(expected_series.max(), actual_series.max())
        breakpoints = np.linspace(minimum, maximum, bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = pd.cut(expected_series, bins=breakpoints, include_lowest=True).value_counts(sort=False)
    actual_counts = pd.cut(actual_series, bins=breakpoints, include_lowest=True).value_counts(sort=False)
    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)
    expected_pct = expected_pct.clip(lower=epsilon)
    actual_pct = actual_pct.clip(lower=epsilon)

    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def feature_drift_report(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    bins: int = 10,
    psi_threshold: float = 0.2,
) -> pd.DataFrame:
    """Return per-feature PSI drift diagnostics."""
    common_columns = [
        column
        for column in train_features.select_dtypes(include="number").columns
        if column in test_features.select_dtypes(include="number").columns
    ]
    rows = []
    for column in common_columns:
        psi = population_stability_index(train_features[column], test_features[column], bins=bins)
        rows.append(
            {
                "feature": column,
                "psi": psi,
                "drift_level": classify_psi(psi, threshold=psi_threshold),
                "drift_alert": bool(psi >= psi_threshold) if not np.isnan(psi) else False,
            },
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def classify_psi(psi: float, threshold: float = 0.2) -> str:
    """Classify PSI into stable, watch, or drift."""
    if np.isnan(psi):
        return "unknown"
    if psi >= threshold:
        return "drift"
    if psi >= threshold / 2:
        return "watch"
    return "stable"


def retrain_triggers(
    rolling_ic: pd.Series,
    ic_threshold: float = 0.02,
    consecutive_periods: int = 1,
) -> pd.DataFrame:
    """Return timestamps where rolling IC decay should trigger retraining."""
    clean_ic = rolling_ic.dropna()
    below_threshold = clean_ic < ic_threshold
    if consecutive_periods <= 1:
        trigger_mask = below_threshold
    else:
        trigger_mask = below_threshold.rolling(consecutive_periods).sum() >= consecutive_periods

    return pd.DataFrame(
        {
            "rolling_ic": clean_ic,
            "below_threshold": below_threshold,
            "retrain_trigger": trigger_mask.fillna(False).astype(bool),
        },
    )


def plot_rolling_ic(rolling_ic: pd.Series, threshold: float = 0.02):
    """Create a Plotly rolling IC chart with a retrain threshold line."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install plotly to plot rolling IC.") from exc

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=rolling_ic.index, y=rolling_ic, mode="lines", name="Rolling IC"))
    figure.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Retrain threshold")
    figure.update_layout(
        title="Rolling Out-of-Sample Information Coefficient",
        xaxis_title="Date",
        yaxis_title="IC",
        template="plotly_white",
    )
    return figure
