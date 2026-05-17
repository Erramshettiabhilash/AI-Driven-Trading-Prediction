"""Regime detection features for market-state-aware models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeDetectorConfig:
    """Configuration for regime detection methods."""

    hmm_states: int = 3
    kmeans_clusters: int = 3
    rolling_vol_window: int = 20
    trend_window: int = 20
    volume_window: int = 20
    adx_window: int = 14
    adx_trending_threshold: float = 25.0
    adx_ranging_threshold: float = 20.0
    random_state: int = 42


class RegimeDetector:
    """Detect market regimes using HMM, K-Means, and ADX rules."""

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        """Create a regime detector."""
        self.config = config or RegimeDetectorConfig()

    def add_regime_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add base features used by clustering-style regime detection.

        Adds:

        - ``regime_rolling_vol``
        - ``regime_trend_strength``
        - ``regime_volume_ratio``
        """
        output = frame.copy()
        if "close" not in output:
            raise KeyError("Regime features require a close column.")

        returns = output["close"].pct_change()
        output["regime_rolling_vol"] = returns.rolling(self.config.rolling_vol_window).std(ddof=0)
        trend_return = output["close"].pct_change(self.config.trend_window).abs()
        output["regime_trend_strength"] = trend_return / output["regime_rolling_vol"].replace(0, np.nan)

        if "volume" in output:
            volume_mean = output["volume"].rolling(self.config.volume_window).mean().shift(1)
            output["regime_volume_ratio"] = output["volume"] / volume_mean
        else:
            output["regime_volume_ratio"] = 1.0

        return output

    def hmm_regime(self, returns: pd.Series) -> pd.Series:
        """Fit a Gaussian HMM to returns and return hidden-state labels."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install hmmlearn to use HMM regime detection.") from exc

        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_returns) < self.config.hmm_states * 5:
            raise ValueError("Not enough return observations for HMM regime detection.")

        model = GaussianHMM(
            n_components=self.config.hmm_states,
            covariance_type="diag",
            n_iter=200,
            random_state=self.config.random_state,
        )
        values = clean_returns.to_numpy().reshape(-1, 1)
        model.fit(values)
        raw_states = model.predict(values)

        state_means = pd.Series(clean_returns.to_numpy()).groupby(raw_states).mean()
        ordered_states = {state: rank for rank, state in enumerate(state_means.sort_values().index)}
        labels = pd.Series(raw_states, index=clean_returns.index).map(ordered_states)
        return labels.map(lambda state: f"hmm_state_{state}").reindex(returns.index)

    def kmeans_regime(
        self,
        frame: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.Series:
        """Cluster volatility, trend, and volume state using K-Means."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install scikit-learn to use K-Means regime detection.") from exc

        enriched = self.add_regime_features(frame)
        regime_columns = columns or [
            "regime_rolling_vol",
            "regime_trend_strength",
            "regime_volume_ratio",
        ]
        features = enriched[regime_columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(features) < self.config.kmeans_clusters:
            raise ValueError("Not enough observations for K-Means regime detection.")

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "kmeans",
                    KMeans(
                        n_clusters=self.config.kmeans_clusters,
                        random_state=self.config.random_state,
                        n_init=10,
                    ),
                ),
            ],
        )
        raw_labels = pipeline.fit_predict(features)
        labeled = features.assign(raw_label=raw_labels)
        ordered = (
            labeled.groupby("raw_label")["regime_rolling_vol"]
            .mean()
            .sort_values()
            .reset_index()["raw_label"]
            .to_dict()
        )
        label_map = {raw_label: rank for rank, raw_label in ordered.items()}
        labels = pd.Series(raw_labels, index=features.index).map(label_map)
        return labels.map(lambda label: f"kmeans_vol_{label}").reindex(frame.index)

    def adx(self, frame: pd.DataFrame) -> pd.Series:
        """Compute Average Directional Index."""
        required = {"high", "low", "close"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"ADX requires columns: {sorted(missing)}")

        high = frame["high"]
        low = frame["low"]
        close = frame["close"]
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=frame.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=frame.index,
        )

        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        alpha = 1 / self.config.adx_window
        atr = true_range.ewm(alpha=alpha, min_periods=self.config.adx_window, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=self.config.adx_window, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=self.config.adx_window, adjust=False).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.ewm(alpha=alpha, min_periods=self.config.adx_window, adjust=False).mean()

    def adx_regime(self, frame: pd.DataFrame) -> pd.Series:
        """Classify market as trending, ranging, or transition using ADX."""
        adx = self.adx(frame)
        labels = pd.Series("transition", index=frame.index, dtype="object")
        labels = labels.mask(adx > self.config.adx_trending_threshold, "trending")
        labels = labels.mask(adx < self.config.adx_ranging_threshold, "ranging")
        labels = labels.where(adx.notna())
        return labels.rename("adx_regime")

    def add_all_regimes(self, frame: pd.DataFrame, include_hmm: bool = False) -> pd.DataFrame:
        """Append K-Means, ADX, and optionally HMM regime labels to a frame."""
        output = self.add_regime_features(frame)
        output["kmeans_regime"] = self.kmeans_regime(output)
        output["adx"] = self.adx(output)
        output["adx_regime"] = self.adx_regime(output)

        if include_hmm:
            returns = output["close"].pct_change()
            output["hmm_regime"] = self.hmm_regime(returns)

        return output
