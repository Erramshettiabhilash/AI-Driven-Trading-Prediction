import numpy as np
import pandas as pd

from features import (
    aggregate_order_book_imbalance,
    anchored_vwap,
    build_microstructure_features,
    classify_trade_direction,
    cumulative_delta,
    delta_divergence,
    liquidity_sweep_detection,
    order_book_imbalance,
    order_flow_pressure_signal,
    volume_delta,
    vwap_deviation,
    vwap_standard_deviation_bands,
)


def sample_ohlcv(length: int = 30) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 09:30", periods=length, freq="5min", tz="UTC", name="timestamp")
    close = pd.Series(np.linspace(100.0, 103.0, length), index=index)
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series(1000.0, index=index)
    volume.iloc[-1] = 3000.0
    frame = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr_14": 1.0,
        },
        index=index,
    )
    return frame


def sample_trades() -> pd.DataFrame:
    index = pd.date_range("2024-01-01 09:30", periods=5, freq="1min", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "price": [101.0, 99.0, 100.5, 100.7, 100.6],
            "bid": [100.0, 99.0, 100.0, 100.5, 100.4],
            "ask": [101.0, 100.0, 101.0, 100.8, 100.7],
            "quantity": [10.0, 5.0, 3.0, 2.0, 4.0],
        },
        index=index,
    )


def test_order_book_imbalance_formula_and_pressure_signal() -> None:
    imbalance = order_book_imbalance(pd.Series([80.0, 20.0]), pd.Series([20.0, 80.0]))
    pressure = order_flow_pressure_signal(imbalance)

    assert np.isclose(imbalance.iloc[0], 0.6)
    assert np.isclose(imbalance.iloc[1], -0.6)
    assert pressure.tolist() == [0, 0]


def test_aggregate_order_book_imbalance_sums_depth_levels() -> None:
    snapshots = pd.DataFrame(
        {
            "bid_qty_1": [60.0],
            "bid_qty_2": [40.0],
            "ask_qty_1": [25.0],
            "ask_qty_2": [25.0],
        },
    )

    imbalance = aggregate_order_book_imbalance(snapshots)

    assert np.isclose(imbalance.iloc[0], (100.0 - 50.0) / 150.0)


def test_trade_direction_volume_delta_and_cumulative_delta() -> None:
    trades = sample_trades()

    direction = classify_trade_direction(trades)
    delta = volume_delta(trades)
    cumulative = cumulative_delta(trades)

    assert direction.iloc[0] == 1.0
    assert direction.iloc[1] == -1.0
    assert delta.iloc[0] == 10.0
    assert delta.iloc[1] == -5.0
    assert cumulative.iloc[-1] == delta.sum()


def test_liquidity_sweep_detection_flags_wick_reversal_with_volume() -> None:
    frame = sample_ohlcv(25)
    prior_high = frame["high"].iloc[:20].max()
    frame.iloc[-1, frame.columns.get_loc("high")] = prior_high + 2.0
    frame.iloc[-1, frame.columns.get_loc("close")] = prior_high - 0.1

    sweeps = liquidity_sweep_detection(frame, lookback=20, volume_multiplier=1.5)

    assert sweeps["bearish_liquidity_sweep"].iloc[-1] == 1
    assert sweeps["bullish_liquidity_sweep"].iloc[-1] == 0


def test_vwap_deviation_and_standard_deviation_bands() -> None:
    frame = sample_ohlcv(10)

    vwap = anchored_vwap(frame)
    deviation = vwap_deviation(frame["close"], vwap, frame["atr_14"])
    bands = vwap_standard_deviation_bands(frame, window=5)

    assert np.isclose(vwap.iloc[0], frame["close"].iloc[0])
    assert np.isfinite(deviation.iloc[-1])
    assert bands.upper_3.iloc[-1] > bands.upper_2.iloc[-1] > bands.upper_1.iloc[-1]
    assert bands.lower_3.iloc[-1] < bands.lower_2.iloc[-1] < bands.lower_1.iloc[-1]


def test_delta_divergence_detects_price_high_without_delta_confirmation() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC")
    close = pd.Series([100, 101, 102, 103, 104, 105], index=index)
    cumulative = pd.Series([0, 10, 20, 30, 40, 25], index=index)

    divergence = delta_divergence(close, cumulative, lookback=3)

    assert divergence.iloc[-1] == -1


def test_build_microstructure_features_combines_ohlcv_trades_and_order_book() -> None:
    ohlcv = sample_ohlcv(30)
    trades = sample_trades()
    order_book = pd.DataFrame(
        {
            "bid_qty_1": [100, 120, 130, 80, 90],
            "ask_qty_1": [40, 50, 60, 100, 120],
        },
        index=trades.index,
    )

    features = build_microstructure_features(
        ohlcv,
        trades=trades,
        order_book=order_book,
        lookback=5,
    )

    expected = {
        "micro_vwap",
        "vwap_upper_1",
        "vwap_lower_1",
        "micro_vwap_deviation",
        "volume_delta",
        "cumulative_delta",
        "delta_divergence",
        "order_book_imbalance",
        "order_flow_pressure",
    }
    assert expected.issubset(features.columns)
    assert np.isfinite(features["micro_vwap"].iloc[-1])
