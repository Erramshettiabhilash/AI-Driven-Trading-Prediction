import numpy as np
import pandas as pd

from features import FeatureEngineer, build_feature_matrix


def sample_ohlcv(length: int = 80) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC", name="timestamp")
    close = pd.Series(np.linspace(100, 120, length), index=index)
    if length >= 35:
        close.iloc[30:35] += np.array([0.0, 2.0, 4.0, 2.0, 0.0])
    high = close + 1.0
    low = close - 1.0
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(1_000.0, index=index)
    if length > 40:
        volume.iloc[40] = 3_000.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_build_all_features_adds_expected_columns() -> None:
    frame = sample_ohlcv()
    output = FeatureEngineer().build_all_features(frame)

    expected_columns = {
        "log_return",
        "return_5",
        "return_autocorr_lag_1",
        "rsi_14",
        "ema_cross",
        "macd",
        "macd_signal",
        "macd_histogram",
        "roc_5",
        "volatility_10",
        "atr_14",
        "volatility_regime",
        "obv",
        "volume_ratio_20",
        "volume_zscore_20",
        "fractal_high_confirmed",
        "fractal_low_confirmed",
        "bullish_liquidity_sweep",
        "bearish_liquidity_sweep",
        "higher_high",
        "lower_low",
        "trend_structure",
        "vwap",
        "vwap_deviation",
    }

    assert expected_columns.issubset(output.columns)


def test_return_features_are_log_returns() -> None:
    frame = sample_ohlcv()
    output = FeatureEngineer().add_return_features(frame)

    expected = np.log(frame["close"].iloc[5] / frame["close"].iloc[0])
    assert np.isclose(output["return_5"].iloc[5], expected)


def test_volume_zscore_uses_prior_window_only() -> None:
    frame = sample_ohlcv(length=25)
    frame["volume"] = list(range(1, 26))
    engineer = FeatureEngineer(volume_window=5)

    output = engineer.add_volume_features(frame)
    timestamp = frame.index[5]
    prior_values = np.array([1, 2, 3, 4, 5], dtype=float)
    expected = (6 - prior_values.mean()) / prior_values.std(ddof=0)

    assert np.isclose(output.loc[timestamp, "volume_zscore_5"], expected)


def test_confirmed_fractal_signal_is_delayed_until_known() -> None:
    index = pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC", name="timestamp")
    frame = pd.DataFrame(
        {
            "open": [1, 2, 3, 2, 1, 2, 3],
            "high": [1, 2, 5, 2, 1, 2, 3],
            "low": [0, 1, 2, 1, 0, 1, 2],
            "close": [1, 2, 3, 2, 1, 2, 3],
            "volume": [100] * 7,
        },
        index=index,
    )

    output = FeatureEngineer(fractal_left_window=2, fractal_right_window=2).add_market_structure_features(frame)

    assert output["fractal_high_confirmed"].iloc[2] == 0
    assert output["fractal_high_confirmed"].iloc[4] == 1


def test_build_feature_matrix_excludes_raw_ohlcv_columns() -> None:
    matrix = build_feature_matrix(sample_ohlcv(), dropna=False)

    for raw_column in ["open", "high", "low", "close", "volume"]:
        assert raw_column not in matrix.columns
    assert "rsi_14" in matrix.columns
