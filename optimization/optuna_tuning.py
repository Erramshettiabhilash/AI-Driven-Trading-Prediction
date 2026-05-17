"""Bayesian hyperparameter optimization for financial ML models.

The objectives in this module optimize validation Information Coefficient
using chronological ``TimeSeriesSplit`` folds. Random cross-validation is not
used because it leaks future market regimes into past training folds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import information_coefficient
from models import (
    LSTMModelConfig,
    LSTMTimeSeriesModel,
    XGBoostFactorModel,
    XGBoostModelConfig,
    create_lstm_sequences,
    subset_sequence_dataset,
)
from models.xgboost_factor import TaskType


@dataclass(frozen=True)
class XGBoostSearchSpace:
    """Optuna search bounds for XGBoost factor models."""

    n_estimators: tuple[int, int] = (100, 1000)
    max_depth: tuple[int, int] = (2, 8)
    learning_rate: tuple[float, float] = (0.005, 0.2)
    subsample: tuple[float, float] = (0.5, 1.0)
    colsample_bytree: tuple[float, float] = (0.5, 1.0)
    reg_alpha: tuple[float, float] = (1e-8, 10.0)
    reg_lambda: tuple[float, float] = (1e-8, 10.0)


@dataclass(frozen=True)
class LSTMSearchSpace:
    """Optuna search bounds for LSTM time-series models."""

    num_layers: tuple[int, int] = (1, 3)
    hidden_size: tuple[int, int] = (32, 256)
    dropout: tuple[float, float] = (0.0, 0.5)
    learning_rate: tuple[float, float] = (1e-5, 1e-2)
    batch_size_choices: tuple[int, ...] = (32, 64, 128)


@dataclass(frozen=True)
class StudyResult:
    """Serializable summary of an Optuna study."""

    best_value: float
    best_params: dict[str, Any]
    n_trials: int
    direction: str = "maximize"


def suggest_xgboost_config(
    trial: Any,
    search_space: XGBoostSearchSpace | None = None,
    random_state: int = 42,
    n_jobs: int = 1,
) -> XGBoostModelConfig:
    """Suggest XGBoost hyperparameters from an Optuna-like trial."""
    space = search_space or XGBoostSearchSpace()
    return XGBoostModelConfig(
        n_estimators=trial.suggest_int("n_estimators", *space.n_estimators),
        max_depth=trial.suggest_int("max_depth", *space.max_depth),
        learning_rate=trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        subsample=trial.suggest_float("subsample", *space.subsample),
        colsample_bytree=trial.suggest_float("colsample_bytree", *space.colsample_bytree),
        reg_alpha=trial.suggest_float("reg_alpha", *space.reg_alpha, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", *space.reg_lambda, log=True),
        random_state=random_state,
        n_jobs=n_jobs,
    )


def suggest_lstm_config(
    trial: Any,
    search_space: LSTMSearchSpace | None = None,
    sequence_length: int = 60,
    epochs: int = 100,
    random_state: int = 42,
) -> LSTMModelConfig:
    """Suggest LSTM hyperparameters from an Optuna-like trial."""
    space = search_space or LSTMSearchSpace()
    return LSTMModelConfig(
        sequence_length=sequence_length,
        num_layers=trial.suggest_int("num_layers", *space.num_layers),
        hidden_units=trial.suggest_int("hidden_size", *space.hidden_size, log=True),
        dropout=trial.suggest_float("dropout", *space.dropout),
        learning_rate=trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        batch_size=trial.suggest_categorical("batch_size", list(space.batch_size_choices)),
        epochs=epochs,
        random_state=random_state,
    )


def time_series_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    test_size: int | None = None,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return purged/embargoed ``TimeSeriesSplit`` indices."""
    try:
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install scikit-learn to use time-series CV optimization.") from exc

    dummy = np.arange(n_samples)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_idx, valid_idx in splitter.split(dummy):
        if purge_bars > 0:
            train_idx = train_idx[:-purge_bars]
        if embargo_bars > 0:
            valid_idx = valid_idx[embargo_bars:]
        if len(train_idx) == 0 or len(valid_idx) < 2:
            continue
        splits.append((train_idx, valid_idx))

    if not splits:
        raise ValueError("No valid TimeSeriesSplit folds remain after purge/embargo.")
    return splits


@dataclass
class XGBoostICObjective:
    """Optuna objective that maximizes mean validation IC for XGBoost."""

    x: pd.DataFrame
    y: pd.Series
    realized_returns: pd.Series | None = None
    task: TaskType = "regression"
    n_splits: int = 5
    test_size: int | None = None
    purge_bars: int = 0
    embargo_bars: int = 0
    search_space: XGBoostSearchSpace | None = None
    random_state: int = 42
    n_jobs: int = 1

    def __call__(self, trial: Any) -> float:
        """Return mean validation IC across time-series folds."""
        config = suggest_xgboost_config(
            trial,
            self.search_space,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        fold_ics: list[float] = []

        for fold, (train_idx, valid_idx) in enumerate(
            time_series_cv_splits(
                len(self.x),
                n_splits=self.n_splits,
                test_size=self.test_size,
                purge_bars=self.purge_bars,
                embargo_bars=self.embargo_bars,
            ),
        ):
            x_train = self.x.iloc[train_idx]
            y_train = self.y.iloc[train_idx]
            x_valid = self.x.iloc[valid_idx]
            y_valid = self.y.iloc[valid_idx]

            model = XGBoostFactorModel(task=self.task, config=config)
            model.fit(x_train, y_train, x_valid, y_valid)
            predictions = model.predict(x_valid)
            signal = predictions if self.task == "regression" else predictions - 0.5
            realized = (
                self.realized_returns.iloc[valid_idx]
                if self.realized_returns is not None
                else y_valid
            )
            fold_ic = information_coefficient(signal, realized)
            fold_ics.append(0.0 if np.isnan(fold_ic) else fold_ic)

            intermediate_value = float(np.mean(fold_ics))
            trial.report(intermediate_value, step=fold)
            if trial.should_prune():
                optuna = _optuna()
                raise optuna.TrialPruned()

        return float(np.mean(fold_ics))


@dataclass
class LSTMICObjective:
    """Optuna objective that maximizes mean validation IC for LSTM models."""

    x: pd.DataFrame
    y: pd.Series
    sequence_length: int = 60
    n_splits: int = 5
    test_size: int | None = None
    search_space: LSTMSearchSpace | None = None
    epochs: int = 100
    random_state: int = 42

    def __call__(self, trial: Any) -> float:
        """Return mean validation IC across sequence ``TimeSeriesSplit`` folds."""
        config = suggest_lstm_config(
            trial,
            self.search_space,
            sequence_length=self.sequence_length,
            epochs=self.epochs,
            random_state=self.random_state,
        )
        dataset = create_lstm_sequences(self.x, self.y, sequence_length=self.sequence_length)
        fold_ics: list[float] = []

        for fold, (train_idx, valid_idx) in enumerate(
            time_series_cv_splits(
                len(dataset.y),
                n_splits=self.n_splits,
                test_size=self.test_size,
            ),
        ):
            train = subset_sequence_dataset(dataset, train_idx)
            validation = subset_sequence_dataset(dataset, valid_idx)
            model = LSTMTimeSeriesModel(config)
            model.fit(train, validation=validation)
            predictions = model.predict(validation)
            realized = pd.Series(validation.y, index=validation.index)
            fold_ic = information_coefficient(predictions, realized)
            fold_ics.append(0.0 if np.isnan(fold_ic) else fold_ic)

            intermediate_value = float(np.mean(fold_ics))
            trial.report(intermediate_value, step=fold)
            if trial.should_prune():
                optuna = _optuna()
                raise optuna.TrialPruned()

        return float(np.mean(fold_ics))


def run_xgboost_study(
    x: pd.DataFrame,
    y: pd.Series,
    realized_returns: pd.Series | None = None,
    n_trials: int = 100,
    objective_kwargs: dict[str, Any] | None = None,
    study_name: str = "xgboost_ic_optimization",
    storage: str | None = None,
    seed: int = 42,
) -> tuple[Any, StudyResult]:
    """Run an Optuna TPE study for XGBoost hyperparameters."""
    optuna = _optuna()
    objective = XGBoostICObjective(
        x=x,
        y=y,
        realized_returns=realized_returns,
        **(objective_kwargs or {}),
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials)
    return study, StudyResult(
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        n_trials=len(study.trials),
    )


def run_lstm_study(
    x: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    objective_kwargs: dict[str, Any] | None = None,
    study_name: str = "lstm_ic_optimization",
    storage: str | None = None,
    seed: int = 42,
) -> tuple[Any, StudyResult]:
    """Run an Optuna TPE study for LSTM hyperparameters."""
    optuna = _optuna()
    objective = LSTMICObjective(x=x, y=y, **(objective_kwargs or {}))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials)
    return study, StudyResult(
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        n_trials=len(study.trials),
    )


def save_study_result(result: StudyResult, output_path: str | Path) -> Path:
    """Save a study summary to JSON."""
    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return path


def _optuna() -> Any:
    """Import Optuna with a clear setup error."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install optuna to run Bayesian optimization studies.") from exc
    return optuna
