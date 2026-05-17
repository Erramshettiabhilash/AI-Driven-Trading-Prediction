import numpy as np
import pandas as pd

from models import build_factor_dataset


def test_build_factor_dataset_selects_numeric_features_and_aligns_target() -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC", name="timestamp")
    frame = pd.DataFrame(
        {
            "open": np.arange(20) + 100,
            "high": np.arange(20) + 101,
            "low": np.arange(20) + 99,
            "close": np.arange(20) + 100,
            "volume": 1_000,
            "feature_numeric": np.arange(20),
            "volatility_regime": ["low"] * 20,
        },
        index=index,
    )

    dataset = build_factor_dataset(frame, target_column="target_return_1")

    assert list(dataset.x.columns) == ["feature_numeric"]
    assert dataset.x.index.equals(dataset.y.index)
    assert dataset.y.index.equals(dataset.realized_returns.index)
    assert dataset.x.index.max() == index[-2]
