"""Run the Step 16 Binance live signal engine.

This script subscribes to Binance kline streams, maintains rolling OHLCV
buffers, updates technical features on closed bars, and emits confidence-
weighted signals when at least one model is supplied.

Example:
    python scripts/run_live_signals.py ^
        --symbols BTCUSDT ETHUSDT ^
        --intervals 1h 15m 5m ^
        --xgboost-model results/models/xgboost.pkl ^
        --entry-interval 5m
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd

from live import (
    BinanceKlineStreamer,
    KlineBar,
    LiveFeatureEngine,
    LiveInferenceEngine,
    LiveSignalConfig,
    MultiTimeframeConfluence,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the live signal runner."""
    parser = argparse.ArgumentParser(description="Run Binance WebSocket live signal engine.")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Binance symbols.")
    parser.add_argument("--intervals", nargs="+", default=["1h", "15m", "5m"], help="Kline intervals.")
    parser.add_argument("--trend-interval", default="1h", help="Trend timeframe.")
    parser.add_argument("--confirmation-interval", default="15m", help="Confirmation timeframe.")
    parser.add_argument("--entry-interval", default="5m", help="Entry timeframe that triggers signals.")
    parser.add_argument("--buffer-bars", type=int, default=200, help="Rolling OHLCV bars per symbol/timeframe.")
    parser.add_argument("--xgboost-model", default=None, help="Optional joblib/pickle XGBoost model path.")
    parser.add_argument("--lstm-model", default=None, help="Optional Keras LSTM model path.")
    parser.add_argument("--regime-model", default=None, help="Optional joblib/pickle regime detector path.")
    parser.add_argument("--feature-columns", nargs="*", default=None, help="Optional model feature columns.")
    parser.add_argument("--threshold", type=float, default=0.001, help="BUY/SELL prediction threshold.")
    parser.add_argument("--confidence-scale", type=float, default=0.01, help="Prediction magnitude for 100% confidence.")
    parser.add_argument("--reconnect-backoff", type=float, default=5.0, help="Reconnect sleep in seconds.")
    parser.add_argument("--max-reconnect-attempts", type=int, default=None, help="Optional reconnect cap.")
    parser.add_argument("--output-jsonl", default=None, help="Optional file for emitted signal JSON lines.")
    return parser.parse_args()


def load_joblib_model(path: str | None) -> Any:
    """Load a joblib/pickle model when a path is supplied."""
    if path is None:
        return None
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("joblib is required to load serialized sklearn/XGBoost models.") from exc
    return joblib.load(path)


def load_keras_model(path: str | None) -> Any:
    """Load a Keras model when a path is supplied."""
    if path is None:
        return None
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("TensorFlow is required to load an LSTM model.") from exc
    return keras.models.load_model(path)


def signal_to_json(signal: Any) -> str:
    """Serialize a live signal dataclass to compact JSON."""
    payload = {
        "symbol": signal.symbol,
        "timestamp": pd.Timestamp(signal.timestamp).isoformat(),
        "signal": signal.signal,
        "confidence": signal.confidence,
        "prediction": signal.prediction,
        "regime": signal.regime,
        "model_predictions": signal.model_predictions,
        "confluence": None,
    }
    if signal.confluence is not None:
        payload["confluence"] = {
            "direction": signal.confluence.direction,
            "confidence": signal.confluence.confidence,
            "votes": signal.confluence.votes,
        }
    return json.dumps(payload, separators=(",", ":"))


async def main() -> None:
    """Start the Binance stream and emit live signal events."""
    args = parse_args()
    feature_engine = LiveFeatureEngine.create(max_bars=args.buffer_bars)
    confluence_engine = MultiTimeframeConfluence(
        trend_interval=args.trend_interval,
        confirmation_interval=args.confirmation_interval,
        entry_interval=args.entry_interval,
    )
    inference_engine = LiveInferenceEngine(
        xgboost_model=load_joblib_model(args.xgboost_model),
        lstm_model=load_keras_model(args.lstm_model),
        regime_detector=load_joblib_model(args.regime_model),
        config=LiveSignalConfig(
            buy_threshold=args.threshold,
            sell_threshold=-args.threshold,
            confidence_scale=args.confidence_scale,
            feature_columns=tuple(args.feature_columns) if args.feature_columns else None,
        ),
    )
    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    async def on_bar(bar: KlineBar) -> None:
        features = feature_engine.update_bar(bar)
        if not bar.is_closed:
            return
        if bar.interval != args.entry_interval:
            return

        frames = {
            interval: feature_engine.compute_features(feature_engine.buffer.get(bar.symbol, interval))
            for interval in args.intervals
        }
        confluence = confluence_engine.evaluate(frames)
        if inference_engine.xgboost_model is None and inference_engine.lstm_model is None:
            print(
                json.dumps(
                    {
                        "symbol": bar.symbol,
                        "timestamp": str(features.index[-1]),
                        "event": "features_updated",
                        "confluence": confluence.votes,
                    },
                ),
            )
            return

        signal = inference_engine.generate_signal(bar.symbol, features, confluence=confluence)
        line = signal_to_json(signal)
        print(line)
        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    streamer = BinanceKlineStreamer(
        symbols=args.symbols,
        intervals=args.intervals,
        on_bar=on_bar,
        reconnect_backoff_seconds=args.reconnect_backoff,
        max_reconnect_attempts=args.max_reconnect_attempts,
    )
    await streamer.run()


if __name__ == "__main__":
    asyncio.run(main())
