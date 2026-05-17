"""Live multi-timeframe confluence and model inference signal engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from evaluation import prediction_to_signal


class PredictiveModel(Protocol):
    """Minimal model protocol used by the live inference engine."""

    def predict(self, features: pd.DataFrame | np.ndarray) -> object:
        """Return one or more numeric predictions."""


class RegimeDetector(Protocol):
    """Minimal regime detector protocol for live routing."""

    def predict(self, features: pd.DataFrame | np.ndarray) -> object:
        """Return one or more regime labels."""


@dataclass(frozen=True)
class MultiTimeframeDecision:
    """Directional vote from H1 trend, M15 confirmation, and M5 entry context."""

    direction: str
    confidence: float
    votes: dict[str, str]


@dataclass(frozen=True)
class LiveSignal:
    """Final live signal emitted by the inference engine."""

    symbol: str
    timestamp: pd.Timestamp
    signal: str
    confidence: float
    prediction: float
    regime: str | None = None
    model_predictions: dict[str, float] = field(default_factory=dict)
    confluence: MultiTimeframeDecision | None = None


@dataclass(frozen=True)
class LiveSignalConfig:
    """Configuration for live confidence-weighted signal generation."""

    buy_threshold: float = 0.001
    sell_threshold: float = -0.001
    confidence_scale: float = 0.01
    blocked_regimes: tuple[str, ...] = ("risk_off", "crash", "halt")
    feature_columns: tuple[str, ...] | None = None
    lstm_sequence_length: int = 60


def _last_scalar_prediction(prediction: object) -> float:
    """Extract the latest scalar prediction from common model outputs."""
    array = np.asarray(prediction, dtype=float)
    if array.size == 0:
        return float("nan")
    return float(array.reshape(-1)[-1])


def _last_label(prediction: object) -> str:
    """Extract the latest label from a model output."""
    array = np.asarray(prediction, dtype=object)
    if array.size == 0:
        return "unknown"
    return str(array.reshape(-1)[-1])


def latest_model_frame(features: pd.DataFrame, feature_columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Return the latest single-row model feature frame."""
    selected = features[list(feature_columns)] if feature_columns is not None else features
    numeric = selected.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("No numeric feature columns are available for live inference.")
    return numeric.tail(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def lstm_sequence_frame(
    features: pd.DataFrame,
    sequence_length: int = 60,
    feature_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return the latest LSTM sequence as a feature frame."""
    selected = features[list(feature_columns)] if feature_columns is not None else features
    numeric = selected.select_dtypes(include=[np.number])
    if len(numeric) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} rows for LSTM inference.")
    return numeric.tail(sequence_length).replace([np.inf, -np.inf], np.nan).fillna(0.0)


class MultiTimeframeConfluence:
    """Evaluate H1 trend, M15 confirmation, and M5 entry alignment."""

    def __init__(
        self,
        trend_interval: str = "1h",
        confirmation_interval: str = "15m",
        entry_interval: str = "5m",
        volume_ratio_column: str = "volume_ratio_20",
    ) -> None:
        self.trend_interval = trend_interval
        self.confirmation_interval = confirmation_interval
        self.entry_interval = entry_interval
        self.volume_ratio_column = volume_ratio_column

    def evaluate(self, feature_frames: dict[str, pd.DataFrame]) -> MultiTimeframeDecision:
        """Return a directional confluence decision across configured timeframes."""
        votes = {
            "trend": self._trend_vote(feature_frames.get(self.trend_interval)),
            "confirmation": self._confirmation_vote(feature_frames.get(self.confirmation_interval)),
            "entry": self._entry_vote(feature_frames.get(self.entry_interval)),
        }
        bullish = sum(vote == "BUY" for vote in votes.values())
        bearish = sum(vote == "SELL" for vote in votes.values())
        total_votes = sum(vote != "HOLD" for vote in votes.values())
        if bullish > bearish:
            direction = "BUY"
            confidence = bullish / len(votes)
        elif bearish > bullish:
            direction = "SELL"
            confidence = bearish / len(votes)
        else:
            direction = "HOLD"
            confidence = 0.0 if total_votes == 0 else 0.5
        return MultiTimeframeDecision(direction=direction, confidence=float(confidence), votes=votes)

    def _trend_vote(self, frame: pd.DataFrame | None) -> str:
        if frame is None or frame.empty or "ema_cross" not in frame:
            return "HOLD"
        value = float(frame["ema_cross"].iloc[-1])
        return "BUY" if value > 0 else "SELL" if value < 0 else "HOLD"

    def _confirmation_vote(self, frame: pd.DataFrame | None) -> str:
        if frame is None or frame.empty:
            return "HOLD"
        latest = frame.iloc[-1]
        rsi = float(latest.get("rsi_14", 50.0))
        macd_histogram = float(latest.get("macd_histogram", 0.0))
        if rsi > 50 and macd_histogram > 0:
            return "BUY"
        if rsi < 50 and macd_histogram < 0:
            return "SELL"
        return "HOLD"

    def _entry_vote(self, frame: pd.DataFrame | None) -> str:
        if frame is None or frame.empty:
            return "HOLD"
        latest = frame.iloc[-1]
        macd_histogram = float(latest.get("macd_histogram", 0.0))
        volume_ratio = float(latest.get(self.volume_ratio_column, 1.0))
        volume_zscore = float(latest.get("volume_zscore_20", 0.0))
        volume_ok = volume_ratio >= 1.0 or volume_zscore > 0.0
        if not volume_ok:
            return "HOLD"
        return "BUY" if macd_histogram > 0 else "SELL" if macd_histogram < 0 else "HOLD"


class LiveInferenceEngine:
    """Combine XGBoost, LSTM, confluence, and regime labels into live signals."""

    def __init__(
        self,
        xgboost_model: PredictiveModel | None = None,
        lstm_model: PredictiveModel | None = None,
        regime_detector: RegimeDetector | None = None,
        config: LiveSignalConfig | None = None,
    ) -> None:
        self.xgboost_model = xgboost_model
        self.lstm_model = lstm_model
        self.regime_detector = regime_detector
        self.config = config or LiveSignalConfig()

    def predict_models(self, features: pd.DataFrame) -> dict[str, float]:
        """Return latest predictions from all configured models."""
        predictions: dict[str, float] = {}
        latest = latest_model_frame(features, self.config.feature_columns)
        if self.xgboost_model is not None:
            predictions["xgboost"] = _last_scalar_prediction(self.xgboost_model.predict(latest))
        if self.lstm_model is not None:
            sequence = lstm_sequence_frame(
                features,
                sequence_length=self.config.lstm_sequence_length,
                feature_columns=self.config.feature_columns,
            )
            try:
                lstm_prediction = self.lstm_model.predict(sequence)
            except Exception:
                lstm_prediction = self.lstm_model.predict(sequence.to_numpy(dtype=float)[None, :, :])
            predictions["lstm"] = _last_scalar_prediction(lstm_prediction)
        return predictions

    def detect_regime(self, features: pd.DataFrame) -> str | None:
        """Detect the latest market regime when a detector is configured."""
        if self.regime_detector is None:
            return None
        latest = latest_model_frame(features, self.config.feature_columns)
        return _last_label(self.regime_detector.predict(latest))

    def generate_signal(
        self,
        symbol: str,
        features: pd.DataFrame,
        confluence: MultiTimeframeDecision | None = None,
    ) -> LiveSignal:
        """Generate a confidence-weighted BUY, SELL, or HOLD signal."""
        model_predictions = self.predict_models(features)
        if not model_predictions:
            raise ValueError("At least one predictive model is required for live signal generation.")

        prediction = float(np.nanmean(list(model_predictions.values())))
        regime = self.detect_regime(features)
        signal = prediction_to_signal(
            pd.Series([prediction]),
            buy_threshold=self.config.buy_threshold,
            sell_threshold=self.config.sell_threshold,
        ).iloc[0]

        confidence = min(abs(prediction) / self.config.confidence_scale, 1.0)
        if confluence is not None:
            if confluence.direction == "HOLD" or confluence.direction != signal:
                confidence *= 0.5
            else:
                confidence *= 0.5 + 0.5 * confluence.confidence

        if regime is not None and regime in self.config.blocked_regimes:
            signal = "HOLD"
            confidence = 0.0

        return LiveSignal(
            symbol=symbol.upper(),
            timestamp=pd.Timestamp(features.index[-1]),
            signal=str(signal),
            confidence=float(confidence),
            prediction=prediction,
            regime=regime,
            model_predictions=model_predictions,
            confluence=confluence,
        )
