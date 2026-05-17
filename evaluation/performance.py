"""Trading performance metrics for model-derived signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def signal_to_position(
    signal: pd.Series | np.ndarray,
    threshold: float = 0.0,
) -> pd.Series:
    """Map prediction scores into long, flat, or short positions."""
    signal_series = pd.Series(signal).astype(float)
    return pd.Series(
        np.select(
            [signal_series > threshold, signal_series < -threshold],
            [1.0, -1.0],
            default=0.0,
        ),
        index=signal_series.index,
        name="position",
    )


def signal_strategy_returns(
    signal: pd.Series | np.ndarray,
    realized_returns: pd.Series | np.ndarray,
    threshold: float = 0.0,
    transaction_cost_bps: float = 0.0,
) -> pd.Series:
    """Convert prediction scores into one-period strategy returns.

    The position at timestamp ``t`` is applied to the forward return label at
    timestamp ``t``. Transaction costs are charged whenever the position
    changes.
    """
    frame = pd.DataFrame({"signal": signal, "realized": realized_returns}).dropna()
    positions = signal_to_position(frame["signal"], threshold=threshold)
    turnover = positions.diff().abs().fillna(positions.abs())
    cost = turnover * transaction_cost_bps / 10_000
    returns = positions * frame["realized"] - cost
    returns.name = "strategy_return"
    return returns


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Return annualized Sharpe ratio for a return series."""
    returns_series = pd.Series(returns).dropna()
    if returns_series.empty:
        return float("nan")
    excess_returns = returns_series - risk_free_rate / annualization_factor
    volatility = excess_returns.std(ddof=0)
    if volatility == 0:
        return float("nan")
    return float(excess_returns.mean() / volatility * np.sqrt(annualization_factor))


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """Return maximum drawdown from a simple return series."""
    returns_series = pd.Series(returns).dropna()
    if returns_series.empty:
        return float("nan")
    equity = (1 + returns_series).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def cagr(returns: pd.Series | np.ndarray, annualization_factor: int = 252) -> float:
    """Return compound annual growth rate from a simple return series."""
    returns_series = pd.Series(returns).dropna()
    if returns_series.empty:
        return float("nan")
    total_return = float((1 + returns_series).prod())
    years = len(returns_series) / annualization_factor
    if years <= 0 or total_return <= 0:
        return float("nan")
    return float(total_return ** (1 / years) - 1)


def calmar_ratio(returns: pd.Series | np.ndarray, annualization_factor: int = 252) -> float:
    """Return CAGR divided by absolute max drawdown."""
    annual_return = cagr(returns, annualization_factor=annualization_factor)
    drawdown = max_drawdown(returns)
    if drawdown == 0 or np.isnan(drawdown):
        return float("nan")
    return float(annual_return / abs(drawdown))


def profit_factor(returns: pd.Series | np.ndarray) -> float:
    """Return gross profits divided by gross losses."""
    returns_series = pd.Series(returns).dropna()
    gross_profit = returns_series[returns_series > 0].sum()
    gross_loss = abs(returns_series[returns_series < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return float(gross_profit / gross_loss)


def win_rate(returns: pd.Series | np.ndarray) -> float:
    """Return fraction of non-zero returns that are positive."""
    returns_series = pd.Series(returns).dropna()
    non_zero = returns_series[returns_series != 0]
    if non_zero.empty:
        return float("nan")
    return float((non_zero > 0).mean())
