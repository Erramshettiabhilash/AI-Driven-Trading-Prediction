"""Incremental feature updates for live OHLCV buffers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.technical import FeatureEngineer
from live.market_stream import KlineBar, RollingOHLCVBuffer


@dataclass
class LiveFeatureEngine:
    """Update rolling buffers and recompute the latest point-in-time features."""

    buffer: RollingOHLCVBuffer
    feature_engineer: FeatureEngineer

    @classmethod
    def create(cls, max_bars: int = 200, feature_engineer: FeatureEngineer | None = None) -> "LiveFeatureEngine":
        """Create a live feature engine with a capped OHLCV buffer."""
        return cls(
            buffer=RollingOHLCVBuffer(max_bars=max_bars),
            feature_engineer=feature_engineer or FeatureEngineer(),
        )

    def update_bar(self, bar: KlineBar) -> pd.DataFrame:
        """Update the buffer with a new bar and return the full feature frame."""
        frame = self.buffer.update(bar)
        return self.compute_features(frame)

    def compute_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute features on the current rolling buffer without dropping rows."""
        if ohlcv.empty:
            return pd.DataFrame()
        clean = ohlcv.drop(columns=["close_time", "is_closed"], errors="ignore")
        return self.feature_engineer.build_all_features(clean)

    def latest_features(self, symbol: str, interval: str) -> pd.Series:
        """Return the latest feature row for a buffered symbol and interval."""
        frame = self.compute_features(self.buffer.get(symbol, interval))
        if frame.empty:
            raise KeyError(f"No features available for {symbol.upper()} {interval}.")
        return frame.iloc[-1]
