"""Preprocessing utilities for point-in-time-safe market research data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
CANONICAL_COLUMN_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "volume": "volume",
}


@dataclass(frozen=True)
class DataPreprocessor:
    """Clean, align, and normalize OHLCV data without lookahead leakage."""

    missing_value_method: str = "ffill"
    outlier_clip_sigma: float = 3.0
    rolling_zscore_window: int = 60
    min_periods: int = 20

    def standardize_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with canonical lowercase OHLCV column names."""
        output = frame.copy()
        output.columns = [
            CANONICAL_COLUMN_MAP.get(str(column).strip().lower(), str(column).strip().lower())
            for column in output.columns
        ]
        return output

    def ensure_utc_index(
        self,
        frame: pd.DataFrame,
        timestamp_column: str | None = None,
    ) -> pd.DataFrame:
        """Return a sorted copy indexed by UTC timestamps.

        Args:
            frame: Input DataFrame with either a ``DatetimeIndex`` or a timestamp
                column.
            timestamp_column: Optional timestamp column to promote into the
                index before timezone alignment.
        """
        output = frame.copy()

        if timestamp_column:
            output[timestamp_column] = pd.to_datetime(output[timestamp_column], utc=True)
            output = output.set_index(timestamp_column)
        elif not isinstance(output.index, pd.DatetimeIndex):
            output.index = pd.to_datetime(output.index, utc=True)
        elif output.index.tz is None:
            output.index = output.index.tz_localize("UTC")
        else:
            output.index = output.index.tz_convert("UTC")

        output.index.name = "timestamp"
        return output.sort_index().loc[~output.index.duplicated(keep="last")]

    def fill_missing_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using the configured point-in-time-safe method."""
        output = frame.copy()

        if self.missing_value_method == "ffill":
            return output.ffill()
        if self.missing_value_method == "zero":
            return output.fillna(0)
        if self.missing_value_method == "drop":
            return output.dropna()

        raise ValueError(f"Unsupported missing value method: {self.missing_value_method}")

    def forward_fill_intraday_gaps(
        self,
        frame: pd.DataFrame,
        frequency: str,
        price_columns: Iterable[str] = PRICE_COLUMNS,
        volume_column: str = "volume",
    ) -> pd.DataFrame:
        """Forward-fill missing intraday bars and mark synthetic rows.

        Missing intraday bars are common with API outages, thin symbols, and
        exchange maintenance windows. This method creates a complete timestamp
        grid, forward-fills price fields using only past values, sets missing
        volume to zero, and adds ``is_synthetic_gap`` so downstream models can
        learn that the row was reconstructed.
        """
        output = self.ensure_utc_index(frame)
        complete_index = pd.date_range(
            start=output.index.min(),
            end=output.index.max(),
            freq=frequency,
            tz="UTC",
            name="timestamp",
        )
        output = output.reindex(complete_index)
        output["is_synthetic_gap"] = output[volume_column].isna() if volume_column in output else True

        existing_price_columns = [column for column in price_columns if column in output.columns]
        output[existing_price_columns] = output[existing_price_columns].ffill()

        if volume_column in output.columns:
            output[volume_column] = output[volume_column].fillna(0.0)

        metadata_columns = [
            column for column in ["symbol", "source", "asset_class"] if column in output.columns
        ]
        output[metadata_columns] = output[metadata_columns].ffill()
        return output

    def clip_return_outliers(
        self,
        frame: pd.DataFrame,
        price_column: str = "close",
        output_column: str = "log_return_clipped",
    ) -> pd.DataFrame:
        """Clip log-return outliers using expanding statistics shifted by one bar.

        The expanding mean and standard deviation are shifted so the clipping
        threshold at time ``t`` is based only on observations available before
        ``t``. This avoids leaking future volatility information into the past.
        """
        if price_column not in frame:
            raise KeyError(f"Missing price column: {price_column}")

        output = frame.copy()
        log_return = np.log(output[price_column] / output[price_column].shift(1))
        expanding_mean = log_return.expanding(min_periods=self.min_periods).mean().shift(1)
        expanding_std = log_return.expanding(min_periods=self.min_periods).std(ddof=0).shift(1)

        lower = expanding_mean - self.outlier_clip_sigma * expanding_std
        upper = expanding_mean + self.outlier_clip_sigma * expanding_std
        output["log_return"] = log_return
        output[output_column] = log_return.clip(lower=lower, upper=upper)

        return output

    def rolling_zscore(
        self,
        frame: pd.DataFrame,
        columns: Iterable[str],
        window: int | None = None,
        suffix: str = "_zscore",
    ) -> pd.DataFrame:
        """Add rolling z-score columns using only prior observations.

        For a feature value at time ``t``, the rolling mean and standard
        deviation are shifted by one bar. This means the normalized feature
        uses the current value but compares it with a distribution known before
        the current bar closed.
        """
        output = frame.copy()
        lookback = window or self.rolling_zscore_window

        for column in columns:
            if column not in output:
                raise KeyError(f"Missing feature column for z-score: {column}")

            rolling_mean = output[column].rolling(lookback, min_periods=self.min_periods).mean().shift(1)
            rolling_std = output[column].rolling(lookback, min_periods=self.min_periods).std(ddof=0).shift(1)
            output[f"{column}{suffix}"] = (output[column] - rolling_mean) / rolling_std.replace(0, np.nan)

        return output

    def clean_ohlcv(
        self,
        frame: pd.DataFrame,
        timestamp_column: str | None = None,
        intraday_frequency: str | None = None,
    ) -> pd.DataFrame:
        """Run the default Step 2 preprocessing chain for one OHLCV dataset."""
        output = self.standardize_columns(frame)
        output = self.ensure_utc_index(output, timestamp_column=timestamp_column)

        if intraday_frequency:
            output = self.forward_fill_intraday_gaps(output, frequency=intraday_frequency)

        output = self.fill_missing_values(output)
        output = self.clip_return_outliers(output)

        numeric_columns = [
            column
            for column in ["open", "high", "low", "close", "volume", "log_return_clipped"]
            if column in output.columns
        ]
        return self.rolling_zscore(output, columns=numeric_columns)


def align_assets_to_utc(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple UTC-indexed asset frames into one aligned panel.

    The returned frame has a ``MultiIndex`` of ``timestamp`` and ``symbol``.
    This form is convenient for cross-asset feature construction while keeping
    every observation tied to the exact point in time when it was known.
    """
    aligned_frames: list[pd.DataFrame] = []
    preprocessor = DataPreprocessor()

    for symbol, frame in frames.items():
        output = preprocessor.ensure_utc_index(frame)
        if "symbol" not in output:
            output["symbol"] = symbol
        aligned_frames.append(output)

    panel = pd.concat(aligned_frames).sort_index()
    panel = panel.reset_index().set_index(["timestamp", "symbol"]).sort_index()
    return panel
