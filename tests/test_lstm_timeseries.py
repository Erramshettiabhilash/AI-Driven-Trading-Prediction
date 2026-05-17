import importlib.util

import numpy as np
import pandas as pd
import pytest

from models import LSTMModelConfig, LSTMTimeSeriesModel, create_lstm_sequences, subset_sequence_dataset
from models.lstm_timeseries import time_series_sequence_splits


def sample_sequence_inputs(length: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC", name="timestamp")
    features = pd.DataFrame(
        {
            "feature_a": np.arange(length, dtype=float),
            "feature_b": np.arange(length, dtype=float) * 2,
        },
        index=index,
    )
    target = pd.Series(np.arange(length, dtype=float) / 100, index=index, name="target_return_1")
    return features, target


def test_create_lstm_sequences_aligns_last_timestep_with_target() -> None:
    features, target = sample_sequence_inputs(length=8)

    dataset = create_lstm_sequences(features, target, sequence_length=3)

    assert dataset.x.shape == (6, 3, 2)
    assert dataset.y.shape == (6,)
    assert dataset.index[0] == features.index[2]
    assert np.isclose(dataset.y[0], target.iloc[2])
    assert dataset.x[0, :, 0].tolist() == [0.0, 1.0, 2.0]


def test_create_lstm_sequences_drops_missing_rows_before_windowing() -> None:
    features, target = sample_sequence_inputs(length=8)
    features.iloc[3, 0] = np.nan

    dataset = create_lstm_sequences(features, target, sequence_length=3)

    assert len(dataset.index) == 5
    assert features.index[3] not in dataset.index


def test_subset_sequence_dataset_preserves_feature_metadata() -> None:
    features, target = sample_sequence_inputs(length=10)
    dataset = create_lstm_sequences(features, target, sequence_length=4)

    subset = subset_sequence_dataset(dataset, np.array([0, 2, 4]))

    assert subset.x.shape == (3, 4, 2)
    assert subset.y.tolist() == dataset.y[[0, 2, 4]].tolist()
    assert subset.feature_names == ["feature_a", "feature_b"]


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_time_series_sequence_splits_are_chronological() -> None:
    features, target = sample_sequence_inputs(length=30)
    dataset = create_lstm_sequences(features, target, sequence_length=5)

    splits = time_series_sequence_splits(dataset, n_splits=3, test_size=4)

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not installed")
def test_lstm_model_builds_expected_architecture() -> None:
    config = LSTMModelConfig(sequence_length=5, hidden_units=8, num_layers=2, epochs=1, batch_size=4)
    model = LSTMTimeSeriesModel(config=config)

    keras_model = model.build(n_features=2)

    lstm_layers = [layer for layer in keras_model.layers if "LSTM" in layer.__class__.__name__]
    dropout_layers = [layer for layer in keras_model.layers if "Dropout" in layer.__class__.__name__]
    assert len(lstm_layers) == 2
    assert len(dropout_layers) == 2
    assert keras_model.input_shape == (None, 5, 2)
