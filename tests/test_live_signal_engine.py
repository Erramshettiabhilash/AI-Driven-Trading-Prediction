import numpy as np
import pandas as pd

from live import (
    KlineBar,
    LiveFeatureEngine,
    LiveInferenceEngine,
    LiveSignalConfig,
    MultiTimeframeConfluence,
    RollingOHLCVBuffer,
    parse_binance_kline_message,
)


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, _features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.array([self.value])


class ConstantRegime:
    def __init__(self, label: str) -> None:
        self.label = label

    def predict(self, _features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.array([self.label])


def make_bar(step: int, symbol: str = "BTCUSDT", interval: str = "5m", close: float = 100.0) -> KlineBar:
    open_time = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=5 * step)
    return KlineBar(
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        close_time=open_time + pd.Timedelta(minutes=5),
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1000.0 + step,
        is_closed=True,
    )


def feature_frame(rows: int = 70) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "log_return": np.linspace(-0.001, 0.001, rows),
            "rsi_14": np.full(rows, 58.0),
            "ema_cross": np.linspace(0.1, 1.0, rows),
            "macd_histogram": np.full(rows, 0.02),
            "atr_14": np.full(rows, 1.5),
            "volume_ratio_20": np.full(rows, 1.2),
            "volume_zscore_20": np.full(rows, 0.5),
        },
        index=index,
    )


def test_parse_binance_kline_message_normalizes_payload() -> None:
    message = {
        "data": {
            "k": {
                "s": "btcusdt",
                "i": "5m",
                "t": 1_704_067_200_000,
                "T": 1_704_067_499_999,
                "o": "100.0",
                "h": "102.0",
                "l": "99.5",
                "c": "101.0",
                "v": "12.5",
                "x": True,
            },
        },
    }

    bar = parse_binance_kline_message(message)

    assert bar.symbol == "BTCUSDT"
    assert bar.interval == "5m"
    assert bar.close == 101.0
    assert bar.open_time.tz is not None
    assert bar.is_closed


def test_rolling_buffer_replaces_duplicate_and_caps_length() -> None:
    buffer = RollingOHLCVBuffer(max_bars=3)
    for step in range(4):
        buffer.update(make_bar(step, close=100.0 + step))
    replacement = make_bar(3, close=200.0)
    frame = buffer.update(replacement)

    assert len(frame) == 3
    assert frame["close"].iloc[-1] == 200.0
    assert buffer.latest("BTCUSDT", "5m")["close"] == 200.0


def test_live_feature_engine_updates_incremental_indicators() -> None:
    engine = LiveFeatureEngine.create(max_bars=80)
    features = pd.DataFrame()
    for step in range(40):
        features = engine.update_bar(make_bar(step, close=100.0 + step * 0.2))

    latest = features.iloc[-1]

    assert {"rsi_14", "ema_cross", "macd_histogram", "atr_14", "volume_zscore_20"}.issubset(features.columns)
    assert np.isfinite(latest["ema_cross"])
    assert np.isfinite(latest["atr_14"])


def test_multi_timeframe_confluence_detects_bullish_alignment() -> None:
    frames = {
        "1h": feature_frame(),
        "15m": feature_frame(),
        "5m": feature_frame(),
    }
    confluence = MultiTimeframeConfluence().evaluate(frames)

    assert confluence.direction == "BUY"
    assert confluence.confidence == 1.0
    assert set(confluence.votes) == {"trend", "confirmation", "entry"}


def test_live_inference_engine_combines_models_and_confluence() -> None:
    features = feature_frame()
    confluence = MultiTimeframeConfluence().evaluate({"1h": features, "15m": features, "5m": features})
    engine = LiveInferenceEngine(
        xgboost_model=ConstantModel(0.004),
        lstm_model=ConstantModel(0.006),
        regime_detector=ConstantRegime("normal"),
        config=LiveSignalConfig(buy_threshold=0.001, sell_threshold=-0.001, confidence_scale=0.01),
    )

    signal = engine.generate_signal("btcusdt", features, confluence=confluence)

    assert signal.signal == "BUY"
    assert signal.prediction == 0.005
    assert signal.confidence == 0.5
    assert signal.regime == "normal"
    assert set(signal.model_predictions) == {"xgboost", "lstm"}


def test_live_inference_engine_blocks_risk_off_regime() -> None:
    features = feature_frame()
    engine = LiveInferenceEngine(
        xgboost_model=ConstantModel(0.01),
        regime_detector=ConstantRegime("risk_off"),
        config=LiveSignalConfig(buy_threshold=0.001, blocked_regimes=("risk_off",)),
    )

    signal = engine.generate_signal("ETHUSDT", features)

    assert signal.signal == "HOLD"
    assert signal.confidence == 0.0
