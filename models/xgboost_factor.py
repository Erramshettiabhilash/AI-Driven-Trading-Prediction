"""XGBoost factor models for tabular quantitative finance features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from evaluation.metrics import hit_rate, information_coefficient
from evaluation.performance import max_drawdown, sharpe_ratio, signal_strategy_returns


TaskType = Literal["regression", "classification"]


@dataclass(frozen=True)
class XGBoostModelConfig:
    """Configuration for XGBoost factor models."""

    n_estimators: int = 500
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42
    tree_method: str = "hist"
    n_jobs: int = -1


@dataclass(frozen=True)
class XGBoostEvaluation:
    """Evaluation summary for an XGBoost model and its derived signal."""

    rmse: float | None
    accuracy: float | None
    ic: float
    hit_rate: float
    signal_sharpe: float
    signal_max_drawdown: float
    best_iteration: int | None


def _rank_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Return Spearman IC without requiring SciPy."""
    frame = pd.DataFrame({"prediction": predictions, "label": labels}).dropna()
    if len(frame) < 2:
        return 0.0
    value = frame["prediction"].rank().corr(frame["label"].rank())
    if pd.isna(value):
        return 0.0
    return float(value)


def _ic_eval_metric(predictions: np.ndarray, dmatrix: object) -> tuple[str, float]:
    """XGBoost custom metric returning validation rank IC."""
    labels = dmatrix.get_label()
    return "ic", _rank_ic(predictions, labels)


class XGBoostFactorModel:
    """Train and evaluate XGBoost return or direction models.

    The wrapper uses XGBoost's native training API so validation IC can be used
    for early stopping. Regression models optimize squared error while early
    stopping selects the boosting round with the best validation rank IC.
    Classification models optimize binary log loss and use validation rank IC
    between predicted probabilities and direction labels.
    """

    def __init__(self, task: TaskType, config: XGBoostModelConfig | None = None) -> None:
        """Initialize the factor model."""
        if task not in {"regression", "classification"}:
            raise ValueError("task must be either 'regression' or 'classification'.")
        self.task = task
        self.config = config or XGBoostModelConfig()
        self.booster_: object | None = None
        self.feature_names_: list[str] = []
        self.best_iteration_: int | None = None

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "XGBoostFactorModel":
        """Fit the model, optionally early-stopping on validation IC."""
        xgb = self._xgboost()
        self.feature_names_ = list(x_train.columns)
        y_train_prepared = self._prepare_target(y_train)

        dtrain = xgb.DMatrix(x_train, label=y_train_prepared, feature_names=self.feature_names_)
        evals = [(dtrain, "train")]

        custom_metric = None
        early_stopping_rounds = None
        if x_valid is not None and y_valid is not None:
            y_valid_prepared = self._prepare_target(y_valid)
            dvalid = xgb.DMatrix(x_valid, label=y_valid_prepared, feature_names=self.feature_names_)
            evals.append((dvalid, "validation"))
            custom_metric = _ic_eval_metric
            early_stopping_rounds = self.config.early_stopping_rounds

        self.booster_ = xgb.train(
            params=self._params(),
            dtrain=dtrain,
            num_boost_round=self.config.n_estimators,
            evals=evals,
            custom_metric=custom_metric,
            maximize=True if custom_metric else None,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        self.best_iteration_ = getattr(self.booster_, "best_iteration", None)
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict returns or positive-direction probabilities."""
        self._check_is_fitted()
        xgb = self._xgboost()
        dmatrix = xgb.DMatrix(x[self.feature_names_], feature_names=self.feature_names_)
        kwargs = {}
        if self.best_iteration_ is not None:
            kwargs["iteration_range"] = (0, self.best_iteration_ + 1)
        predictions = self.booster_.predict(dmatrix, **kwargs)
        name = "predicted_return" if self.task == "regression" else "predicted_up_probability"
        return pd.Series(predictions, index=x.index, name=name)

    def predict_direction(self, x: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Predict ``1`` or ``-1`` direction labels for classification models."""
        if self.task != "classification":
            raise ValueError("predict_direction is available only for classification models.")
        probabilities = self.predict(x)
        return pd.Series(np.where(probabilities >= threshold, 1, -1), index=x.index, name="direction")

    def evaluate(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        realized_returns: pd.Series | None = None,
        signal_threshold: float = 0.0,
        transaction_cost_bps: float = 0.0,
        annualization_factor: int = 252,
    ) -> XGBoostEvaluation:
        """Evaluate predictions with statistical and trading metrics."""
        predictions = self.predict(x)
        realized = realized_returns if realized_returns is not None else y

        rmse = None
        accuracy = None
        signal = predictions

        if self.task == "regression":
            rmse = float(np.sqrt(np.mean((predictions.loc[y.index] - y) ** 2)))
        else:
            labels = self._direction_to_pm_one(y)
            directions = pd.Series(
                np.where(predictions.loc[y.index] >= 0.5, 1, -1),
                index=y.index,
                name="predicted_direction",
            )
            accuracy = float((directions == labels).mean())
            signal = predictions - 0.5

        strategy_returns = signal_strategy_returns(
            signal=signal,
            realized_returns=realized,
            threshold=signal_threshold,
            transaction_cost_bps=transaction_cost_bps,
        )

        return XGBoostEvaluation(
            rmse=rmse,
            accuracy=accuracy,
            ic=information_coefficient(signal, realized),
            hit_rate=hit_rate(signal, realized),
            signal_sharpe=sharpe_ratio(strategy_returns, annualization_factor=annualization_factor),
            signal_max_drawdown=max_drawdown(strategy_returns),
            best_iteration=self.best_iteration_,
        )

    def save_model(self, path: str | Path) -> None:
        """Save the trained XGBoost booster to disk."""
        self._check_is_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(output_path)

    def load_model(self, path: str | Path, feature_names: list[str]) -> "XGBoostFactorModel":
        """Load a trained XGBoost booster from disk."""
        xgb = self._xgboost()
        self.booster_ = xgb.Booster()
        self.booster_.load_model(path)
        self.feature_names_ = feature_names
        self.best_iteration_ = None
        return self

    def _params(self) -> dict[str, object]:
        """Return native XGBoost parameters for the configured task."""
        objective = "reg:squarederror" if self.task == "regression" else "binary:logistic"
        eval_metric = "rmse" if self.task == "regression" else "logloss"
        return {
            "objective": objective,
            "eval_metric": eval_metric,
            "max_depth": self.config.max_depth,
            "eta": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "alpha": self.config.reg_alpha,
            "lambda": self.config.reg_lambda,
            "seed": self.config.random_state,
            "tree_method": self.config.tree_method,
            "nthread": self.config.n_jobs,
        }

    def _prepare_target(self, y: pd.Series) -> pd.Series:
        """Prepare target labels for XGBoost training."""
        if self.task == "classification":
            return self._direction_to_zero_one(y)
        return y.astype(float)

    @staticmethod
    def _direction_to_zero_one(y: pd.Series) -> pd.Series:
        """Map direction labels from ``{-1, 1}`` or ``{0, 1}`` into ``{0, 1}``."""
        unique_values = set(pd.Series(y).dropna().unique())
        if unique_values.issubset({0, 1}):
            return y.astype(int)
        if unique_values.issubset({-1, 1}):
            return pd.Series(np.where(y > 0, 1, 0), index=y.index)
        raise ValueError("Classification target must contain either {-1, 1} or {0, 1}.")

    @staticmethod
    def _direction_to_pm_one(y: pd.Series) -> pd.Series:
        """Map direction labels from ``{0, 1}`` or ``{-1, 1}`` into ``{-1, 1}``."""
        unique_values = set(pd.Series(y).dropna().unique())
        if unique_values.issubset({-1, 1}):
            return y.astype(int)
        if unique_values.issubset({0, 1}):
            return pd.Series(np.where(y > 0, 1, -1), index=y.index)
        raise ValueError("Classification target must contain either {-1, 1} or {0, 1}.")

    @staticmethod
    def _xgboost() -> object:
        """Import XGBoost with a clear installation error."""
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install xgboost to use XGBoostFactorModel.") from exc
        return xgb

    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted or loaded."""
        if self.booster_ is None:
            raise RuntimeError("The XGBoostFactorModel is not fitted.")
