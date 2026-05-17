"""Trading integration and portfolio evaluation utilities.

This module turns model predictions into portfolio returns with position
sizing, allocation, and execution costs. It is intentionally deterministic so
research backtests can be reproduced before the same logic is reused in live
trading components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.metrics import information_coefficient
from evaluation.performance import (
    cagr,
    calmar_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)


@dataclass(frozen=True)
class ExecutionCostConfig:
    """Execution friction assumptions expressed in basis points."""

    slippage_bps: float = 2.0
    spread_bps: float = 1.0
    commission_bps: float = 0.0


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Container for an execution-aware portfolio backtest."""

    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    equity_curve: pd.Series
    target_weights: pd.DataFrame
    turnover: pd.Series
    metrics: dict[str, float]


def prediction_to_signal(
    predictions: pd.Series | np.ndarray,
    buy_threshold: float,
    sell_threshold: float | None = None,
) -> pd.Series:
    """Map prediction scores to BUY, SELL, and HOLD labels.

    Thresholds matter because tiny model scores are often noise. In trading,
    acting on every non-zero prediction can create turnover that overwhelms
    alpha after slippage and spread costs.
    """
    prediction_series = pd.Series(predictions).astype(float)
    sell_level = -buy_threshold if sell_threshold is None else sell_threshold
    return pd.Series(
        np.select(
            [prediction_series > buy_threshold, prediction_series < sell_level],
            ["BUY", "SELL"],
            default="HOLD",
        ),
        index=prediction_series.index,
        name="signal",
    )


def prediction_to_position(
    predictions: pd.Series | np.ndarray,
    threshold: float,
    allow_short: bool = True,
) -> pd.Series:
    """Map prediction scores to numeric long, short, or flat positions."""
    signals = prediction_to_signal(predictions, buy_threshold=threshold)
    sell_value = -1.0 if allow_short else 0.0
    return signals.map({"BUY": 1.0, "SELL": sell_value, "HOLD": 0.0}).rename("position")


def atr_position_size(
    account_value: float,
    atr: pd.Series | np.ndarray,
    price: pd.Series | np.ndarray,
    risk_per_trade: float = 0.01,
    atr_multiplier: float = 2.0,
    max_position_weight: float = 0.05,
) -> pd.Series:
    """Return ATR-based units using Position Size = Account_Risk / Stop_Distance.

    Args:
        account_value: Total account equity in currency units.
        atr: Average True Range in price units.
        price: Current asset price.
        risk_per_trade: Fraction of equity to risk if the stop is hit.
        atr_multiplier: Stop distance in ATR multiples.
        max_position_weight: Maximum notional allocation to this position.

    Returns:
        Number of units allowed after both stop-risk and position-weight caps.
    """
    atr_series = pd.Series(atr).astype(float)
    price_series = pd.Series(price, index=atr_series.index).astype(float)
    stop_distance = (atr_series * atr_multiplier).replace(0, np.nan)
    account_risk = account_value * risk_per_trade
    risk_units = account_risk / stop_distance
    max_units = account_value * max_position_weight / price_series.replace(0, np.nan)
    units = pd.concat([risk_units, max_units], axis=1).min(axis=1).replace([np.inf, -np.inf], np.nan)
    return units.fillna(0.0).clip(lower=0.0).rename("position_size_units")


def signal_strength_weights(
    predictions: pd.DataFrame,
    threshold: float = 0.0,
    max_position_weight: float = 0.05,
    max_gross_leverage: float = 1.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Convert cross-asset prediction scores into capped portfolio weights.

    Stronger absolute scores receive larger weights, but each position is capped
    and total gross exposure is constrained. This makes allocation depend on
    signal strength instead of treating all model outputs as equally reliable.
    """
    scores = predictions.astype(float).copy()
    signed_scores = scores.where(scores.abs() > threshold, 0.0)
    if not allow_short:
        signed_scores = signed_scores.clip(lower=0.0)

    absolute_sum = signed_scores.abs().sum(axis=1).replace(0, np.nan)
    weights = signed_scores.div(absolute_sum, axis=0).fillna(0.0) * max_gross_leverage
    weights = weights.clip(lower=-max_position_weight if allow_short else 0.0, upper=max_position_weight)

    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    leverage_scale = (max_gross_leverage / gross).clip(upper=1.0).fillna(0.0)
    return weights.mul(leverage_scale, axis=0)


def execution_costs(
    turnover: pd.Series,
    cost_config: ExecutionCostConfig | None = None,
) -> pd.Series:
    """Return per-period execution costs from turnover and basis-point costs."""
    config = cost_config or ExecutionCostConfig()
    one_way_cost_bps = config.slippage_bps + config.spread_bps / 2 + config.commission_bps
    return (turnover.astype(float) * one_way_cost_bps / 10_000).rename("execution_cost")


def active_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization_factor: int = 252,
) -> float:
    """Return annualized information ratio versus a benchmark return stream."""
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return float("nan")
    active_returns = aligned["strategy"] - aligned["benchmark"]
    tracking_error = active_returns.std(ddof=0)
    if tracking_error == 0:
        return float("nan")
    return float(active_returns.mean() / tracking_error * np.sqrt(annualization_factor))


def portfolio_performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    predictions: pd.Series | None = None,
    realized_returns: pd.Series | None = None,
    annualization_factor: int = 252,
) -> dict[str, float]:
    """Return the Step 15 portfolio metric set."""
    summary = {
        "sharpe_ratio": sharpe_ratio(returns, annualization_factor=annualization_factor),
        "max_drawdown": max_drawdown(returns),
        "cagr": cagr(returns, annualization_factor=annualization_factor),
        "calmar_ratio": calmar_ratio(returns, annualization_factor=annualization_factor),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
        "information_ratio": float("nan"),
        "information_coefficient": float("nan"),
    }
    if benchmark_returns is not None:
        summary["information_ratio"] = active_information_ratio(
            returns,
            benchmark_returns,
            annualization_factor=annualization_factor,
        )
    if predictions is not None and realized_returns is not None:
        summary["information_coefficient"] = information_coefficient(predictions, realized_returns)
    return summary


def backtest_signal_portfolio(
    predictions: pd.DataFrame,
    asset_returns: pd.DataFrame,
    threshold: float = 0.0,
    max_position_weight: float = 0.05,
    max_gross_leverage: float = 1.0,
    allow_short: bool = True,
    cost_config: ExecutionCostConfig | None = None,
    benchmark_returns: pd.Series | None = None,
    annualization_factor: int = 252,
) -> PortfolioBacktestResult:
    """Backtest signal-strength-weighted allocation with execution costs.

    Predictions and returns are aligned by timestamp and asset column. The
    prediction at timestamp ``t`` is assumed to forecast the return label at the
    same timestamp, matching the project's target construction convention.
    """
    common_columns = [column for column in predictions.columns if column in asset_returns.columns]
    if not common_columns:
        raise ValueError("Predictions and asset returns have no overlapping asset columns.")

    frame_index = predictions.index.intersection(asset_returns.index)
    if frame_index.empty:
        raise ValueError("Predictions and asset returns have no overlapping timestamps.")

    aligned_predictions = predictions.loc[frame_index, common_columns].astype(float)
    aligned_returns = asset_returns.loc[frame_index, common_columns].astype(float)
    valid_rows = aligned_predictions.notna().any(axis=1) & aligned_returns.notna().any(axis=1)
    aligned_predictions = aligned_predictions.loc[valid_rows].fillna(0.0)
    aligned_returns = aligned_returns.loc[valid_rows].fillna(0.0)

    weights = signal_strength_weights(
        aligned_predictions,
        threshold=threshold,
        max_position_weight=max_position_weight,
        max_gross_leverage=max_gross_leverage,
        allow_short=allow_short,
    )
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1)).rename("turnover")
    costs = execution_costs(turnover, cost_config)
    gross_returns = (weights * aligned_returns).sum(axis=1).rename("gross_return")
    net_returns = (gross_returns - costs).rename("strategy_return")
    equity_curve = (1 + net_returns).cumprod().rename("equity_curve")

    portfolio_prediction = (weights * aligned_predictions).sum(axis=1)
    opportunity_return = (weights * aligned_returns).sum(axis=1)
    metrics = portfolio_performance_summary(
        net_returns,
        benchmark_returns=benchmark_returns,
        predictions=portfolio_prediction,
        realized_returns=opportunity_return,
        annualization_factor=annualization_factor,
    )
    metrics["average_turnover"] = float(turnover.mean())
    metrics["total_cost"] = float(costs.sum())
    metrics["final_equity"] = float(equity_curve.iloc[-1]) if not equity_curve.empty else float("nan")

    return PortfolioBacktestResult(
        returns=net_returns,
        gross_returns=gross_returns,
        costs=costs,
        equity_curve=equity_curve,
        target_weights=weights,
        turnover=turnover,
        metrics=metrics,
    )
