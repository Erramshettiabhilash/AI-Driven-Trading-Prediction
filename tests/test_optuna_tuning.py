import numpy as np
import pandas as pd

from optimization import (
    LSTMSearchSpace,
    XGBoostICObjective,
    XGBoostSearchSpace,
    suggest_lstm_config,
    suggest_xgboost_config,
    time_series_cv_splits,
)


class FakeTrial:
    def __init__(self) -> None:
        self.reports: list[tuple[float, int]] = []

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        values = {
            "n_estimators": 20,
            "max_depth": 2,
            "num_layers": 2,
            "hidden_size": 32,
        }
        return values.get(name, low)

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        values = {
            "learning_rate": 0.1 if high > 0.1 else 0.001,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "dropout": 0.2,
        }
        return values.get(name, low)

    def suggest_categorical(self, name: str, choices: list[int]) -> int:
        return choices[0]

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return False


def sample_optimization_data(length: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC", name="timestamp")
    feature_a = np.sin(np.arange(length) / 5)
    feature_b = np.cos(np.arange(length) / 9)
    x = pd.DataFrame({"feature_a": feature_a, "feature_b": feature_b}, index=index)
    y = pd.Series(0.01 * feature_a - 0.005 * feature_b, index=index, name="target_return_1")
    return x, y


def test_suggest_xgboost_config_uses_expected_fields() -> None:
    config = suggest_xgboost_config(
        FakeTrial(),
        XGBoostSearchSpace(n_estimators=(20, 20), max_depth=(2, 2)),
        n_jobs=1,
    )

    assert config.n_estimators == 20
    assert config.max_depth == 2
    assert config.n_jobs == 1


def test_suggest_lstm_config_uses_expected_fields() -> None:
    config = suggest_lstm_config(
        FakeTrial(),
        LSTMSearchSpace(batch_size_choices=(16, 32)),
        sequence_length=30,
        epochs=5,
    )

    assert config.sequence_length == 30
    assert config.num_layers == 2
    assert config.batch_size == 16
    assert config.epochs == 5


def test_time_series_cv_splits_respect_purge_and_embargo() -> None:
    splits = time_series_cv_splits(
        n_samples=30,
        n_splits=3,
        test_size=5,
        purge_bars=2,
        embargo_bars=1,
    )

    assert len(splits) == 3
    for train_idx, valid_idx in splits:
        assert train_idx.max() < valid_idx.min()
        assert valid_idx.min() - train_idx.max() >= 3


def test_xgboost_ic_objective_returns_mean_validation_ic() -> None:
    x, y = sample_optimization_data()
    objective = XGBoostICObjective(
        x=x,
        y=y,
        n_splits=3,
        test_size=20,
        search_space=XGBoostSearchSpace(
            n_estimators=(20, 20),
            max_depth=(2, 2),
            learning_rate=(0.1, 0.1),
            subsample=(0.8, 0.8),
            colsample_bytree=(0.8, 0.8),
            reg_alpha=(0.01, 0.01),
            reg_lambda=(1.0, 1.0),
        ),
        n_jobs=1,
    )
    trial = FakeTrial()

    score = objective(trial)

    assert score > 0.5
    assert len(trial.reports) == 3
