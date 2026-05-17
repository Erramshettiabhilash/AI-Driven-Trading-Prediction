"""Research metrics for predictive trading models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(
    predictions: pd.Series | np.ndarray,
    realized_returns: pd.Series | np.ndarray,
    method: str = "spearman",
) -> float:
    """Return the correlation between predictions and realized returns.

    IC measures whether the model ranks future returns correctly. Spearman rank
    correlation is the default because quant signals often care more about
    ordering than exact magnitude.
    """
    frame = pd.DataFrame({"prediction": predictions, "realized": realized_returns}).dropna()
    if len(frame) < 2:
        return float("nan")
    if method == "spearman":
        return float(frame["prediction"].rank().corr(frame["realized"].rank()))
    return float(frame["prediction"].corr(frame["realized"], method=method))


def rolling_information_coefficient(
    predictions: pd.Series,
    realized_returns: pd.Series,
    window: int = 20,
    method: str = "spearman",
) -> pd.Series:
    """Return rolling IC over a fixed lookback window."""
    frame = pd.DataFrame({"prediction": predictions, "realized": realized_returns}).dropna()
    if method == "spearman":
        values = []
        index = []
        for end in range(window, len(frame) + 1):
            sample = frame.iloc[end - window : end]
            values.append(information_coefficient(sample["prediction"], sample["realized"], method=method))
            index.append(sample.index[-1])
        return pd.Series(values, index=index, name="rolling_ic")
    return frame["prediction"].rolling(window).corr(frame["realized"], method=method)


def information_ratio(
    information_coefficients: pd.Series | np.ndarray,
    annualization_factor: int = 252,
) -> float:
    """Annualize mean IC by IC volatility."""
    ic = pd.Series(information_coefficients).dropna()
    if ic.empty:
        return float("nan")
    ic_std = ic.std(ddof=0)
    if ic_std == 0:
        return float("nan")
    return float(ic.mean() / ic_std * np.sqrt(annualization_factor))


def hit_rate(
    predictions: pd.Series | np.ndarray,
    realized_returns: pd.Series | np.ndarray,
) -> float:
    """Return the fraction of observations with matching predicted/realized sign."""
    frame = pd.DataFrame({"prediction": predictions, "realized": realized_returns}).dropna()
    if frame.empty:
        return float("nan")

    predicted_sign = np.sign(frame["prediction"])
    realized_sign = np.sign(frame["realized"])
    non_zero = (predicted_sign != 0) & (realized_sign != 0)
    if not non_zero.any():
        return float("nan")
    return float((predicted_sign[non_zero] == realized_sign[non_zero]).mean())
