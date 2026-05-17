"""LSTM sequence models for financial time-series forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import hit_rate, information_coefficient
from evaluation.performance import max_drawdown, sharpe_ratio, signal_strategy_returns


@dataclass(frozen=True)
class SequenceDataset:
    """Three-dimensional sequence dataset for recurrent neural networks."""

    x: np.ndarray
    y: np.ndarray
    index: pd.Index
    feature_names: list[str]


@dataclass(frozen=True)
class LSTMModelConfig:
    """Configuration for the Step 6 LSTM forecasting model."""

    sequence_length: int = 60
    hidden_units: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    random_state: int = 42


@dataclass(frozen=True)
class LSTMEvaluation:
    """Evaluation summary for LSTM forecasts and the derived trading signal."""

    rmse: float
    ic: float
    hit_rate: float
    signal_sharpe: float
    signal_max_drawdown: float


def create_lstm_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int = 60,
) -> SequenceDataset:
    """Create ``T x N`` LSTM samples from aligned tabular features and labels.

    The sample ending at timestamp ``t`` contains feature rows
    ``[t - sequence_length + 1, ..., t]`` and is paired with the target label
    at ``t``. This preserves the same prediction timestamp semantics used by
    the XGBoost factor model.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    frame = features.join(target.rename("target"), how="inner").dropna()
    if len(frame) < sequence_length:
        raise ValueError("Not enough rows to create one LSTM sequence.")

    feature_names = list(features.columns)
    feature_values = frame[feature_names].to_numpy(dtype=np.float32)
    target_values = frame["target"].to_numpy(dtype=np.float32)

    sequences: list[np.ndarray] = []
    labels: list[float] = []
    index: list[Any] = []

    for end_position in range(sequence_length - 1, len(frame)):
        start_position = end_position - sequence_length + 1
        sequences.append(feature_values[start_position : end_position + 1])
        labels.append(float(target_values[end_position]))
        index.append(frame.index[end_position])

    return SequenceDataset(
        x=np.asarray(sequences, dtype=np.float32),
        y=np.asarray(labels, dtype=np.float32),
        index=pd.Index(index, name=features.index.name),
        feature_names=feature_names,
    )


def time_series_sequence_splits(
    dataset: SequenceDataset,
    n_splits: int = 5,
    test_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return scikit-learn ``TimeSeriesSplit`` indices for sequence data."""
    try:
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install scikit-learn to use TimeSeriesSplit validation.") from exc

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(dataset.x)]


def subset_sequence_dataset(dataset: SequenceDataset, indices: np.ndarray) -> SequenceDataset:
    """Return a row subset of a ``SequenceDataset`` while preserving metadata."""
    return SequenceDataset(
        x=dataset.x[indices],
        y=dataset.y[indices],
        index=dataset.index[indices],
        feature_names=dataset.feature_names,
    )


class LSTMTimeSeriesModel:
    """Two-layer LSTM model for next-period return forecasting."""

    def __init__(self, config: LSTMModelConfig | None = None) -> None:
        """Initialize an unbuilt LSTM model wrapper."""
        self.config = config or LSTMModelConfig()
        self.model_: Any | None = None
        self.feature_names_: list[str] = []

    def build(self, n_features: int) -> Any:
        """Build and compile the Keras LSTM architecture."""
        keras = self._keras()
        tf = self._tensorflow()
        tf.random.set_seed(self.config.random_state)

        model = keras.Sequential(name="lstm_return_forecaster")
        model.add(
            keras.layers.Input(
                shape=(self.config.sequence_length, n_features),
                name="feature_sequence",
            ),
        )

        for layer_number in range(self.config.num_layers):
            return_sequences = layer_number < self.config.num_layers - 1
            model.add(
                keras.layers.LSTM(
                    self.config.hidden_units,
                    return_sequences=return_sequences,
                    name=f"lstm_{layer_number + 1}",
                ),
            )
            model.add(keras.layers.Dropout(self.config.dropout, name=f"dropout_{layer_number + 1}"))

        model.add(keras.layers.Dense(1, name="predicted_return"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        self.model_ = model
        return model

    def fit(
        self,
        train: SequenceDataset,
        validation: SequenceDataset | None = None,
        verbose: int = 0,
    ) -> Any:
        """Fit the LSTM with early stopping and ReduceLROnPlateau."""
        keras = self._keras()
        self.feature_names_ = train.feature_names

        if self.model_ is None:
            self.build(n_features=train.x.shape[-1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation else "loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if validation else "loss",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_learning_rate,
            ),
        ]
        validation_data = (validation.x, validation.y) if validation else None
        return self.model_.fit(
            train.x,
            train.y,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            shuffle=False,
            verbose=verbose,
        )

    def predict(self, dataset: SequenceDataset) -> pd.Series:
        """Predict returns for a sequence dataset."""
        self._check_is_fitted()
        predictions = self.model_.predict(dataset.x, verbose=0).reshape(-1)
        return pd.Series(predictions, index=dataset.index, name="predicted_return")

    def evaluate(
        self,
        dataset: SequenceDataset,
        realized_returns: pd.Series | None = None,
        signal_threshold: float = 0.0,
        transaction_cost_bps: float = 0.0,
        annualization_factor: int = 252,
    ) -> LSTMEvaluation:
        """Evaluate LSTM predictions with statistical and trading metrics."""
        predictions = self.predict(dataset)
        target = pd.Series(dataset.y, index=dataset.index, name="target")
        realized = realized_returns.reindex(dataset.index) if realized_returns is not None else target
        strategy_returns = signal_strategy_returns(
            predictions,
            realized,
            threshold=signal_threshold,
            transaction_cost_bps=transaction_cost_bps,
        )

        return LSTMEvaluation(
            rmse=float(np.sqrt(np.mean((predictions - target) ** 2))),
            ic=information_coefficient(predictions, realized),
            hit_rate=hit_rate(predictions, realized),
            signal_sharpe=sharpe_ratio(strategy_returns, annualization_factor=annualization_factor),
            signal_max_drawdown=max_drawdown(strategy_returns),
        )

    def save_model(self, path: str | Path) -> None:
        """Save the Keras model to disk."""
        self._check_is_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save(output_path)

    def load_model(self, path: str | Path, feature_names: list[str]) -> "LSTMTimeSeriesModel":
        """Load a saved Keras model."""
        keras = self._keras()
        self.model_ = keras.models.load_model(path)
        self.feature_names_ = feature_names
        return self

    @staticmethod
    def _tensorflow() -> Any:
        """Import TensorFlow with a clear setup error."""
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install tensorflow to train LSTMTimeSeriesModel.") from exc
        return tf

    @staticmethod
    def _keras() -> Any:
        """Return TensorFlow Keras."""
        tf = LSTMTimeSeriesModel._tensorflow()
        return tf.keras

    def _check_is_fitted(self) -> None:
        """Raise if the Keras model has not been built or trained."""
        if self.model_ is None:
            raise RuntimeError("The LSTMTimeSeriesModel is not fitted.")
