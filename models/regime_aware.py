"""Regime-aware model routing for market-state-dependent alpha."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from evaluation.metrics import hit_rate, information_coefficient
from evaluation.performance import max_drawdown, sharpe_ratio, signal_strategy_returns
from models.xgboost_factor import TaskType, XGBoostEvaluation, XGBoostFactorModel, XGBoostModelConfig


@dataclass
class RegimeAwareXGBoostModel:
    """Train one XGBoost model per regime and route inference dynamically."""

    task: TaskType = "regression"
    model_config: XGBoostModelConfig = field(default_factory=XGBoostModelConfig)
    min_regime_samples: int = 30
    fallback_model: XGBoostFactorModel | None = None
    regime_models: dict[str, XGBoostFactorModel] = field(default_factory=dict)

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        train_regimes: pd.Series,
        x_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        valid_regimes: pd.Series | None = None,
    ) -> "RegimeAwareXGBoostModel":
        """Fit fallback and per-regime XGBoost models."""
        train_regimes = train_regimes.reindex(x_train.index).astype("object")
        self.fallback_model = XGBoostFactorModel(self.task, self.model_config).fit(
            x_train,
            y_train,
            x_valid,
            y_valid,
        )
        self.regime_models = {}

        for regime in sorted(train_regimes.dropna().unique()):
            mask = train_regimes.eq(regime)
            if int(mask.sum()) < self.min_regime_samples:
                continue

            regime_x = x_train.loc[mask]
            regime_y = y_train.loc[mask]
            regime_valid_x = None
            regime_valid_y = None

            if x_valid is not None and y_valid is not None and valid_regimes is not None:
                valid_mask = valid_regimes.reindex(x_valid.index).eq(regime)
                if int(valid_mask.sum()) >= 2:
                    regime_valid_x = x_valid.loc[valid_mask]
                    regime_valid_y = y_valid.loc[valid_mask]

            self.regime_models[str(regime)] = XGBoostFactorModel(self.task, self.model_config).fit(
                regime_x,
                regime_y,
                regime_valid_x,
                regime_valid_y,
            )

        return self

    def predict(self, x: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        """Predict by routing each row to its detected regime model."""
        self._check_is_fitted()
        regimes = regimes.reindex(x.index).astype("object")
        predictions = pd.Series(index=x.index, dtype="float64", name="regime_prediction")

        for regime in regimes.dropna().unique():
            mask = regimes.eq(regime)
            model = self.regime_models.get(str(regime), self.fallback_model)
            predictions.loc[mask] = model.predict(x.loc[mask])

        missing_mask = predictions.isna()
        if missing_mask.any():
            predictions.loc[missing_mask] = self.fallback_model.predict(x.loc[missing_mask])

        return predictions

    def evaluate(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
        realized_returns: pd.Series | None = None,
        signal_threshold: float = 0.0,
        transaction_cost_bps: float = 0.0,
        annualization_factor: int = 252,
    ) -> XGBoostEvaluation:
        """Evaluate routed predictions with Step 5 metrics."""
        predictions = self.predict(x, regimes)
        realized = realized_returns if realized_returns is not None else y
        signal = predictions if self.task == "regression" else predictions - 0.5
        strategy_returns = signal_strategy_returns(
            signal,
            realized,
            threshold=signal_threshold,
            transaction_cost_bps=transaction_cost_bps,
        )

        rmse = None
        accuracy = None
        if self.task == "regression":
            rmse = float(((predictions.loc[y.index] - y) ** 2).mean() ** 0.5)
        else:
            expected = XGBoostFactorModel._direction_to_pm_one(y)
            predicted = pd.Series(
                [1 if value >= 0.5 else -1 for value in predictions.loc[y.index]],
                index=y.index,
            )
            accuracy = float((predicted == expected).mean())

        return XGBoostEvaluation(
            rmse=rmse,
            accuracy=accuracy,
            ic=information_coefficient(signal, realized),
            hit_rate=hit_rate(signal, realized),
            signal_sharpe=sharpe_ratio(strategy_returns, annualization_factor=annualization_factor),
            signal_max_drawdown=max_drawdown(strategy_returns),
            best_iteration=None,
        )

    def model_for_regime(self, regime: str | None) -> XGBoostFactorModel:
        """Return the specific regime model, falling back to the global model."""
        self._check_is_fitted()
        return self.regime_models.get(str(regime), self.fallback_model)

    def _check_is_fitted(self) -> None:
        """Raise if the router has not been fitted."""
        if self.fallback_model is None:
            raise RuntimeError("RegimeAwareXGBoostModel is not fitted.")
