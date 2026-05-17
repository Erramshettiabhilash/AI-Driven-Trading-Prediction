"""Technical feature engineering for quantitative trading research.

Every feature in this module is designed to be computable from information
available at or before the feature timestamp. When a popular indicator normally
uses future bars, such as centered fractals, the implementation delays the
signal until it would have been confirmed in real time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class FeatureEngineer:
    """Create point-in-time-safe price, volume, and market-structure features."""

    return_windows: tuple[int, ...] = (5, 10, 20, 60)
    autocorrelation_lags: tuple[int, ...] = (1, 2, 3, 4, 5)
    rsi_window: int = 14
    ema_fast: int = 8
    ema_slow: int = 33
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_window: int = 14
    volatility_windows: tuple[int, ...] = (10, 20)
    volume_window: int = 20
    fractal_left_window: int = 2
    fractal_right_window: int = 2
    liquidity_lookback: int = 20
    liquidity_volume_multiplier: float = 1.5
    trend_lookback: int = 20
    feature_columns_: list[str] = field(default_factory=list, init=False, repr=False)

    def build_all_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with the complete Step 3 feature set.

        Args:
            frame: OHLCV DataFrame indexed by timestamp. Required columns are
                ``open``, ``high``, ``low``, ``close``, and ``volume``.

        Returns:
            A copy of ``frame`` with return, momentum, volatility, volume, and
            market-structure features appended.
        """
        self._validate_ohlcv(frame)
        output = frame.copy().sort_index()
        output = self.add_return_features(output)
        output = self.add_momentum_features(output)
        output = self.add_volatility_features(output)
        output = self.add_volume_features(output)
        output = self.add_market_structure_features(output)
        return output

    def add_return_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add log returns, rolling returns, and return autocorrelation."""
        output = frame.copy()
        output["log_return"] = np.log(output["close"] / output["close"].shift(1))

        for window in self.return_windows:
            output[f"return_{window}"] = np.log(output["close"] / output["close"].shift(window))

        for lag in self.autocorrelation_lags:
            output[f"return_autocorr_lag_{lag}"] = (
                output["log_return"]
                .rolling(self.volume_window, min_periods=max(5, lag + 2))
                .corr(output["log_return"].shift(lag))
            )

        return output

    def add_momentum_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add RSI, EMA crossover, MACD, and rate-of-change features."""
        output = frame.copy()

        output[f"rsi_{self.rsi_window}"] = self._rsi(output["close"], self.rsi_window)
        ema_fast = output["close"].ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = output["close"].ewm(span=self.ema_slow, adjust=False).mean()
        output[f"ema_{self.ema_fast}"] = ema_fast
        output[f"ema_{self.ema_slow}"] = ema_slow
        output["ema_cross"] = ema_fast - ema_slow
        output["ema_cross_signal"] = np.sign(output["ema_cross"])

        macd_line = output["close"].ewm(span=self.macd_fast, adjust=False).mean() - output[
            "close"
        ].ewm(span=self.macd_slow, adjust=False).mean()
        macd_signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        output["macd"] = macd_line
        output["macd_signal"] = macd_signal_line
        output["macd_histogram"] = macd_line - macd_signal_line

        for window in self.return_windows:
            output[f"roc_{window}"] = output["close"].pct_change(window)

        return output

    def add_volatility_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add rolling volatility, ATR, and volatility regime labels."""
        output = frame.copy()

        if "log_return" not in output:
            output["log_return"] = np.log(output["close"] / output["close"].shift(1))

        for window in self.volatility_windows:
            output[f"volatility_{window}"] = output["log_return"].rolling(window).std(ddof=0)

        output[f"atr_{self.atr_window}"] = self._atr(output, self.atr_window)
        output["volatility_regime"] = self._volatility_regime(output[f"volatility_{self.volatility_windows[-1]}"])

        return output

    def add_volume_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add OBV, volume ratio versus moving average, and volume z-score."""
        output = frame.copy()
        signed_volume = np.sign(output["close"].diff()).fillna(0.0) * output["volume"]
        output["obv"] = signed_volume.cumsum()

        rolling_volume_mean = output["volume"].rolling(self.volume_window).mean().shift(1)
        rolling_volume_std = output["volume"].rolling(self.volume_window).std(ddof=0).shift(1)
        output[f"volume_ratio_{self.volume_window}"] = output["volume"] / rolling_volume_mean
        output[f"volume_zscore_{self.volume_window}"] = (
            output["volume"] - rolling_volume_mean
        ) / rolling_volume_std.replace(0, np.nan)

        return output

    def add_market_structure_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add confirmed fractals, sweeps, trend structure, and VWAP deviation."""
        output = frame.copy()
        output = self._add_confirmed_fractals(output)

        previous_high = output["high"].rolling(self.liquidity_lookback).max().shift(1)
        previous_low = output["low"].rolling(self.liquidity_lookback).min().shift(1)
        volume_mean = output["volume"].rolling(self.volume_window).mean().shift(1)
        volume_spike = output["volume"] > volume_mean * self.liquidity_volume_multiplier

        output["bullish_liquidity_sweep"] = (
            (output["low"] < previous_low) & (output["close"] > previous_low) & volume_spike
        ).astype(int)
        output["bearish_liquidity_sweep"] = (
            (output["high"] > previous_high) & (output["close"] < previous_high) & volume_spike
        ).astype(int)

        trailing_high = output["high"].rolling(self.trend_lookback).max().shift(1)
        trailing_low = output["low"].rolling(self.trend_lookback).min().shift(1)
        output["higher_high"] = (output["high"] > trailing_high).astype(int)
        output["lower_low"] = (output["low"] < trailing_low).astype(int)
        output["trend_structure"] = np.select(
            [output["higher_high"].eq(1), output["lower_low"].eq(1)],
            [1, -1],
            default=0,
        )

        output["vwap"] = self._anchored_vwap(output)
        atr_column = f"atr_{self.atr_window}"
        if atr_column not in output:
            output[atr_column] = self._atr(output, self.atr_window)
        output["vwap_deviation"] = (output["close"] - output["vwap"]) / output[atr_column].replace(
            0,
            np.nan,
        )

        return output

    def feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """Return engineered feature columns, excluding raw OHLCV and metadata."""
        excluded = {
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "symbol",
            "source",
            "asset_class",
            "is_synthetic_gap",
        }
        return [column for column in frame.columns if column not in excluded]

    def _validate_ohlcv(self, frame: pd.DataFrame) -> None:
        """Raise a helpful error if the OHLCV schema is incomplete."""
        missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing required OHLCV columns: {missing}")

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        """Compute Wilder-style Relative Strength Index."""
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        average_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        average_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        relative_strength = average_gain / average_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + relative_strength))
        return rsi.fillna(50.0)

    @staticmethod
    def _atr(frame: pd.DataFrame, window: int) -> pd.Series:
        """Compute Average True Range using Wilder smoothing."""
        previous_close = frame["close"].shift(1)
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous_close).abs(),
                (frame["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    @staticmethod
    def _volatility_regime(volatility: pd.Series) -> pd.Series:
        """Classify volatility into low, medium, and high expanding quantile regimes."""
        low_quantile = volatility.expanding(min_periods=20).quantile(0.33).shift(1)
        high_quantile = volatility.expanding(min_periods=20).quantile(0.66).shift(1)
        return pd.Series(
            np.select(
                [volatility <= low_quantile, volatility >= high_quantile],
                ["low", "high"],
                default="medium",
            ),
            index=volatility.index,
            name="volatility_regime",
        )

    def _add_confirmed_fractals(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add delayed fractal signals that are known only after confirmation bars."""
        output = frame.copy()
        left = self.fractal_left_window
        right = self.fractal_right_window
        window = left + right + 1

        centered_high = output["high"].rolling(window, center=True).max()
        centered_low = output["low"].rolling(window, center=True).min()
        raw_high = output["high"].eq(centered_high)
        raw_low = output["low"].eq(centered_low)

        output["fractal_high_confirmed"] = raw_high.shift(right).fillna(False).astype(int)
        output["fractal_low_confirmed"] = raw_low.shift(right).fillna(False).astype(int)
        return output

    @staticmethod
    def _anchored_vwap(frame: pd.DataFrame) -> pd.Series:
        """Compute VWAP, resetting daily for intraday data when possible."""
        typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3
        dollar_volume = typical_price * frame["volume"]

        if isinstance(frame.index, pd.DatetimeIndex):
            session = frame.index.tz_convert("UTC").date if frame.index.tz is not None else frame.index.date
            cumulative_dollar_volume = dollar_volume.groupby(session).cumsum()
            cumulative_volume = frame["volume"].groupby(session).cumsum()
        else:
            cumulative_dollar_volume = dollar_volume.cumsum()
            cumulative_volume = frame["volume"].cumsum()

        return cumulative_dollar_volume / cumulative_volume.replace(0, np.nan)


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_engineer: FeatureEngineer | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the Step 3 feature matrix from a single OHLCV frame."""
    engineer = feature_engineer or FeatureEngineer()
    features = engineer.build_all_features(frame)
    feature_columns = engineer.feature_columns(features)
    matrix = features[feature_columns]
    return matrix.dropna() if dropna else matrix
