"""Ensemble framework for combining predictive trading models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import information_coefficient
from evaluation.performance import max_drawdown, sharpe_ratio, signal_strategy_returns


@dataclass(frozen=True)
class EnsembleEvaluation:
    """Evaluation summary comparing individual models and ensemble signals."""

    metrics: pd.DataFrame
    ensemble_prediction: pd.Series
    ensemble_returns: pd.Series


class LinearRegressionBaseline:
    """Simple statistical baseline for return prediction."""

    def __init__(self) -> None:
        """Initialize an unfitted linear regression baseline."""
        self.model_: Any | None = None
        self.feature_names_: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "LinearRegressionBaseline":
        """Fit an ordinary least-squares baseline."""
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install scikit-learn to use LinearRegressionBaseline.") from exc

        self.feature_names_ = list(x.columns)
        self.model_ = LinearRegression()
        self.model_.fit(x[self.feature_names_], y)
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict returns from the fitted linear baseline."""
        if self.model_ is None:
            raise RuntimeError("LinearRegressionBaseline is not fitted.")
        predictions = self.model_.predict(x[self.feature_names_])
        return pd.Series(predictions, index=x.index, name="linear_baseline")


def align_prediction_frame(
    predictions: pd.DataFrame,
    realized_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align model predictions and realized returns on shared timestamps."""
    frame = predictions.join(realized_returns.rename("realized_return"), how="inner").dropna()
    if frame.empty:
        raise ValueError("No overlapping non-null predictions and realized returns.")
    return frame[predictions.columns], frame["realized_return"]


def rolling_model_ic(
    predictions: pd.DataFrame,
    realized_returns: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """Compute shifted rolling IC for each model prediction column.

    The IC available at timestamp ``t`` is calculated from the prior ``window``
    rows only, so ensemble weights at ``t`` do not use the outcome at ``t``.
    """
    aligned_predictions, aligned_realized = align_prediction_frame(predictions, realized_returns)
    ic_frame = pd.DataFrame(index=aligned_predictions.index, columns=aligned_predictions.columns, dtype=float)

    for end in range(window, len(aligned_predictions)):
        history_slice = slice(end - window, end)
        timestamp = aligned_predictions.index[end]
        for column in aligned_predictions.columns:
            ic_frame.loc[timestamp, column] = information_coefficient(
                aligned_predictions[column].iloc[history_slice],
                aligned_realized.iloc[history_slice],
            )

    return ic_frame


def normalize_weights(raw_weights: pd.Series, allow_short_weights: bool = True) -> pd.Series:
    """Normalize model weights, falling back to equal weights when needed."""
    weights = raw_weights.replace([np.inf, -np.inf], np.nan).dropna()
    if weights.empty:
        return pd.Series(1.0 / len(raw_weights), index=raw_weights.index)

    if not allow_short_weights:
        weights = weights.clip(lower=0)
        denominator = weights.sum()
    else:
        denominator = weights.abs().sum()

    if denominator == 0 or np.isnan(denominator):
        return pd.Series(1.0 / len(raw_weights), index=raw_weights.index)

    normalized = weights / denominator
    return normalized.reindex(raw_weights.index).fillna(0.0)


def ic_weighted_ensemble(
    predictions: pd.DataFrame,
    realized_returns: pd.Series,
    window: int = 20,
    allow_short_weights: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """Combine model predictions using shifted rolling IC weights."""
    aligned_predictions, aligned_realized = align_prediction_frame(predictions, realized_returns)
    ic_frame = rolling_model_ic(aligned_predictions, aligned_realized, window=window)
    weights = pd.DataFrame(index=aligned_predictions.index, columns=aligned_predictions.columns, dtype=float)
    ensemble = pd.Series(index=aligned_predictions.index, dtype=float, name="ic_weighted_ensemble")

    for timestamp in aligned_predictions.index:
        raw_weights = ic_frame.loc[timestamp]
        weights.loc[timestamp] = normalize_weights(
            raw_weights,
            allow_short_weights=allow_short_weights,
        )
        ensemble.loc[timestamp] = float((aligned_predictions.loc[timestamp] * weights.loc[timestamp]).sum())

    return ensemble, weights


class RidgeStackingEnsemble:
    """Ridge meta-learner trained on model predictions."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize a Ridge stacking ensemble."""
        self.alpha = alpha
        self.model_: Any | None = None
        self.model_names_: list[str] = []

    def fit(self, predictions: pd.DataFrame, realized_returns: pd.Series) -> "RidgeStackingEnsemble":
        """Fit the Ridge meta-learner on base model predictions."""
        try:
            from sklearn.linear_model import Ridge
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install scikit-learn to use RidgeStackingEnsemble.") from exc

        aligned_predictions, aligned_realized = align_prediction_frame(predictions, realized_returns)
        self.model_names_ = list(aligned_predictions.columns)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(aligned_predictions[self.model_names_], aligned_realized)
        return self

    def predict(self, predictions: pd.DataFrame) -> pd.Series:
        """Predict stacked ensemble returns from base model predictions."""
        if self.model_ is None:
            raise RuntimeError("RidgeStackingEnsemble is not fitted.")
        stacked = self.model_.predict(predictions[self.model_names_].dropna())
        index = predictions[self.model_names_].dropna().index
        return pd.Series(stacked, index=index, name="ridge_stacking_ensemble")


def regime_conditional_weights(
    predictions: pd.DataFrame,
    realized_returns: pd.Series,
    regimes: pd.Series,
    allow_short_weights: bool = True,
) -> pd.DataFrame:
    """Estimate one set of model weights per regime using in-sample IC."""
    aligned_predictions, aligned_realized = align_prediction_frame(predictions, realized_returns)
    aligned_regimes = regimes.reindex(aligned_predictions.index)
    rows: list[pd.Series] = []

    for regime in sorted(aligned_regimes.dropna().unique()):
        mask = aligned_regimes.eq(regime)
        raw = pd.Series(
            {
                column: information_coefficient(aligned_predictions.loc[mask, column], aligned_realized.loc[mask])
                for column in aligned_predictions.columns
            },
            name=str(regime),
        )
        rows.append(normalize_weights(raw, allow_short_weights=allow_short_weights).rename(str(regime)))

    if not rows:
        raise ValueError("No non-null regimes available for regime-conditional weights.")
    return pd.DataFrame(rows)


def regime_conditional_ensemble(
    predictions: pd.DataFrame,
    regimes: pd.Series,
    regime_weights: pd.DataFrame,
) -> pd.Series:
    """Combine predictions using weights selected by the current regime."""
    aligned_regimes = regimes.reindex(predictions.index)
    ensemble = pd.Series(index=predictions.index, dtype=float, name="regime_conditional_ensemble")
    fallback_weights = regime_weights.mean(axis=0)
    fallback_weights = normalize_weights(fallback_weights)

    for timestamp in predictions.index:
        regime = str(aligned_regimes.loc[timestamp])
        weights = regime_weights.loc[regime] if regime in regime_weights.index else fallback_weights
        weights = weights.reindex(predictions.columns).fillna(0.0)
        ensemble.loc[timestamp] = float((predictions.loc[timestamp] * weights).sum())

    return ensemble


def evaluate_prediction_signals(
    predictions: pd.DataFrame,
    realized_returns: pd.Series,
    ensemble_prediction: pd.Series,
    ensemble_name: str = "ensemble",
    transaction_cost_bps: float = 0.0,
    annualization_factor: int = 252,
) -> EnsembleEvaluation:
    """Evaluate individual model signals and the ensemble signal."""
    aligned_predictions, aligned_realized = align_prediction_frame(predictions, realized_returns)
    ensemble = ensemble_prediction.reindex(aligned_predictions.index).dropna()
    aligned_realized = aligned_realized.reindex(ensemble.index)
    aligned_predictions = aligned_predictions.reindex(ensemble.index)

    rows: list[dict[str, float | str]] = []
    all_signals = dict(aligned_predictions.items())
    all_signals[ensemble_name] = ensemble

    ensemble_returns = pd.Series(dtype=float)
    for name, signal in all_signals.items():
        strategy_returns = signal_strategy_returns(
            signal=signal,
            realized_returns=aligned_realized,
            transaction_cost_bps=transaction_cost_bps,
        )
        if name == ensemble_name:
            ensemble_returns = strategy_returns
        rows.append(
            {
                "model": name,
                "ic": information_coefficient(signal, aligned_realized),
                "sharpe": sharpe_ratio(strategy_returns, annualization_factor=annualization_factor),
                "max_drawdown": max_drawdown(strategy_returns),
            },
        )

    return EnsembleEvaluation(
        metrics=pd.DataFrame(rows),
        ensemble_prediction=ensemble,
        ensemble_returns=ensemble_returns,
    )
