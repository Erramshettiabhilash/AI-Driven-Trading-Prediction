import importlib.util

import numpy as np
import pandas as pd
import pytest

from features import RegimeDetector, RegimeDetectorConfig
from models import RegimeAwareXGBoostModel, XGBoostModelConfig


def sample_regime_frame(length: int = 120) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=length, freq="D", tz="UTC", name="timestamp")
    first = np.linspace(100, 120, length // 2)
    second = 120 + np.sin(np.arange(length - length // 2)) * 2
    close = pd.Series(np.concatenate([first, second]), index=index)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(1_000.0 + np.arange(length) * 3, index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_add_regime_features_creates_core_inputs() -> None:
    detector = RegimeDetector()
    output = detector.add_regime_features(sample_regime_frame())

    assert {"regime_rolling_vol", "regime_trend_strength", "regime_volume_ratio"}.issubset(
        output.columns,
    )
    assert output["regime_rolling_vol"].dropna().ge(0).all()


def test_adx_regime_labels_trending_and_ranging_states() -> None:
    config = RegimeDetectorConfig(adx_trending_threshold=15.0, adx_ranging_threshold=10.0)
    detector = RegimeDetector(config)
    labels = detector.adx_regime(sample_regime_frame(length=140))

    non_null = labels.dropna()
    assert set(non_null.unique()).issubset({"trending", "ranging", "transition"})
    assert "trending" in set(non_null.unique())


def test_kmeans_regime_returns_aligned_labels() -> None:
    detector = RegimeDetector(RegimeDetectorConfig(kmeans_clusters=3))
    frame = sample_regime_frame()

    labels = detector.kmeans_regime(frame)

    assert labels.index.equals(frame.index)
    assert labels.dropna().str.startswith("kmeans_vol_").all()
    assert labels.dropna().nunique() <= 3


@pytest.mark.skipif(importlib.util.find_spec("hmmlearn") is None, reason="hmmlearn not installed")
def test_hmm_regime_returns_hidden_state_labels() -> None:
    detector = RegimeDetector(RegimeDetectorConfig(hmm_states=2))
    returns = sample_regime_frame()["close"].pct_change()

    labels = detector.hmm_regime(returns)

    assert labels.index.equals(returns.index)
    assert labels.dropna().str.startswith("hmm_state_").all()


def test_regime_aware_xgboost_routes_predictions_by_regime() -> None:
    index = pd.date_range("2023-01-01", periods=90, freq="D", tz="UTC", name="timestamp")
    x = pd.DataFrame(
        {
            "feature_a": np.sin(np.arange(90) / 4),
            "feature_b": np.cos(np.arange(90) / 7),
        },
        index=index,
    )
    regimes = pd.Series(["trend"] * 45 + ["range"] * 45, index=index)
    y = pd.Series(
        np.where(regimes.eq("trend"), x["feature_a"] * 0.01, -x["feature_b"] * 0.01),
        index=index,
    )
    model = RegimeAwareXGBoostModel(
        task="regression",
        model_config=XGBoostModelConfig(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.1,
            early_stopping_rounds=5,
            n_jobs=1,
        ),
        min_regime_samples=20,
    )

    model.fit(x.iloc[:70], y.iloc[:70], regimes.iloc[:70], x.iloc[70:80], y.iloc[70:80], regimes.iloc[70:80])
    predictions = model.predict(x.iloc[80:], regimes.iloc[80:])

    assert len(predictions) == 10
    assert set(model.regime_models).issubset({"trend", "range"})
    assert predictions.notna().all()
