import numpy as np
import pandas as pd

from evaluation import (
    ExpandingWindowSplit,
    RollingWindowSplit,
    hit_rate,
    information_coefficient,
    information_ratio,
    temporal_train_test_split,
)
from features import TargetBuilder, build_supervised_frame


def sample_price_frame(length: int = 100) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC", name="timestamp")
    close = pd.Series(np.linspace(100, 150, length), index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000,
            "feature_a": np.arange(length, dtype=float),
        },
        index=index,
    )


def test_forward_log_return_target_uses_future_close() -> None:
    frame = sample_price_frame(length=10)
    output = TargetBuilder(horizons=(1, 5), min_history_for_tertiles=3).add_targets(frame)

    assert np.isclose(output["target_return_1"].iloc[0], np.log(frame["close"].iloc[1] / frame["close"].iloc[0]))
    assert np.isclose(output["target_return_5"].iloc[0], np.log(frame["close"].iloc[5] / frame["close"].iloc[0]))
    assert pd.isna(output["target_return_5"].iloc[-1])


def test_direction_target_outputs_plus_or_minus_one() -> None:
    returns = pd.Series([0.01, -0.02, 0.0, np.nan])
    labels = TargetBuilder.direction_label(returns)

    assert labels.tolist()[:3] == [1.0, -1.0, 1.0]
    assert pd.isna(labels.iloc[-1])


def test_build_supervised_frame_aligns_features_and_target() -> None:
    frame = sample_price_frame(length=20)
    x, y = build_supervised_frame(
        frame,
        target_builder=TargetBuilder(horizons=(1,), min_history_for_tertiles=3),
        feature_columns=["feature_a"],
        target_column="target_return_1",
    )

    assert list(x.columns) == ["feature_a"]
    assert len(x) == len(y)
    assert x.index.equals(y.index)
    assert y.index.max() == frame.index[-2]


def test_temporal_train_test_split_respects_purge_and_embargo() -> None:
    data = sample_price_frame(length=20)
    train_idx, test_idx = temporal_train_test_split(data, test_size=5, purge_bars=2, embargo_bars=1)

    assert train_idx[-1] == 12
    assert test_idx[0] == 16
    assert test_idx[-1] == 19


def test_expanding_window_split_grows_training_set() -> None:
    data = sample_price_frame(length=30)
    splits = list(ExpandingWindowSplit(initial_train_size=10, test_size=5).split(data))

    assert len(splits) == 4
    assert len(splits[0][0]) == 10
    assert len(splits[1][0]) == 15
    assert len(splits[0][1]) == 5


def test_rolling_window_split_keeps_training_size_fixed() -> None:
    data = sample_price_frame(length=30)
    splits = list(RollingWindowSplit(train_size=10, test_size=5).split(data))

    assert len(splits) == 4
    assert len(splits[0][0]) == 10
    assert len(splits[1][0]) == 10
    assert splits[1][0][0] == 5


def test_information_metrics() -> None:
    predictions = pd.Series([0.3, 0.2, -0.1, -0.4])
    realized = pd.Series([0.03, 0.01, -0.02, -0.05])
    rolling_ic = pd.Series([0.1, 0.2, 0.3])

    assert information_coefficient(predictions, realized) > 0.9
    assert hit_rate(predictions, realized) == 1.0
    assert information_ratio(rolling_ic) > 0
