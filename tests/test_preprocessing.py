import numpy as np
import pandas as pd

from data.preprocessing import DataPreprocessor, align_assets_to_utc


def test_ensure_utc_index_converts_naive_timestamps() -> None:
    frame = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-01 09:30", "2024-01-01 09:31"]),
    )

    output = DataPreprocessor().ensure_utc_index(frame)

    assert str(output.index.tz) == "UTC"
    assert output.index.name == "timestamp"


def test_forward_fill_intraday_gaps_marks_synthetic_rows() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 102.0],
            "high": [101.0, 103.0],
            "low": [99.0, 101.0],
            "close": [100.5, 102.5],
            "volume": [10.0, 20.0],
            "symbol": ["TEST", "TEST"],
        },
        index=pd.to_datetime(["2024-01-01 09:30Z", "2024-01-01 09:32Z"]),
    )

    output = DataPreprocessor().forward_fill_intraday_gaps(frame, frequency="1min")

    assert len(output) == 3
    assert bool(output.loc[pd.Timestamp("2024-01-01 09:31Z"), "is_synthetic_gap"])
    assert output.loc[pd.Timestamp("2024-01-01 09:31Z"), "close"] == 100.5
    assert output.loc[pd.Timestamp("2024-01-01 09:31Z"), "volume"] == 0.0


def test_rolling_zscore_uses_prior_window_only() -> None:
    frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 100.0]})
    preprocessor = DataPreprocessor(min_periods=2)

    output = preprocessor.rolling_zscore(frame, columns=["feature"], window=2)

    expected_t3 = (3.0 - np.mean([1.0, 2.0])) / np.std([1.0, 2.0])
    assert np.isclose(output.loc[2, "feature_zscore"], expected_t3)


def test_clip_return_outliers_uses_shifted_expanding_threshold() -> None:
    close = [100.0, 101.0, 102.0, 103.0, 104.0, 200.0]
    frame = pd.DataFrame({"close": close})
    preprocessor = DataPreprocessor(min_periods=3, outlier_clip_sigma=1.0)

    output = preprocessor.clip_return_outliers(frame)

    assert "log_return" in output
    assert "log_return_clipped" in output
    assert output.loc[5, "log_return_clipped"] < output.loc[5, "log_return"]


def test_align_assets_to_utc_returns_timestamp_symbol_panel() -> None:
    frames = {
        "AAA": pd.DataFrame(
            {"close": [1.0]},
            index=pd.to_datetime(["2024-01-01 00:00:00"]),
        ),
        "BBB": pd.DataFrame(
            {"close": [2.0]},
            index=pd.to_datetime(["2024-01-01 00:00:00Z"]),
        ),
    }

    output = align_assets_to_utc(frames)

    assert output.index.names == ["timestamp", "symbol"]
    assert set(output.index.get_level_values("symbol")) == {"AAA", "BBB"}
