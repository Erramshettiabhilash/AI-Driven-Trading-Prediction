import numpy as np
import pandas as pd
import pytest

from evaluation import max_drawdown, sharpe_ratio, signal_strategy_returns
from models import XGBoostFactorModel, XGBoostModelConfig

xgboost = pytest.importorskip("xgboost")


def sample_supervised_data(length: int = 180) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = pd.date_range("2022-01-01", periods=length, freq="D", tz="UTC", name="timestamp")
    feature_a = np.sin(np.arange(length) / 5)
    feature_b = np.cos(np.arange(length) / 11)
    realized = 0.01 * feature_a - 0.004 * feature_b + np.linspace(-0.001, 0.001, length)
    direction = pd.Series(np.where(realized >= 0, 1, -1), index=index, name="direction")
    x = pd.DataFrame({"feature_a": feature_a, "feature_b": feature_b}, index=index)
    y = pd.Series(realized, index=index, name="target_return_1")
    return x, y, direction


def small_config() -> XGBoostModelConfig:
    return XGBoostModelConfig(
        n_estimators=40,
        max_depth=2,
        learning_rate=0.1,
        early_stopping_rounds=5,
        n_jobs=1,
    )


def test_xgboost_regressor_trains_predicts_and_evaluates() -> None:
    x, y, _ = sample_supervised_data()
    model = XGBoostFactorModel(task="regression", config=small_config())

    model.fit(x.iloc[:120], y.iloc[:120], x.iloc[120:150], y.iloc[120:150])
    predictions = model.predict(x.iloc[150:])
    evaluation = model.evaluate(x.iloc[150:], y.iloc[150:])

    assert len(predictions) == len(x.iloc[150:])
    assert evaluation.rmse is not None
    assert evaluation.ic > 0.5
    assert evaluation.best_iteration is not None


def test_xgboost_classifier_trains_predicts_and_evaluates_against_returns() -> None:
    x, y, direction = sample_supervised_data()
    model = XGBoostFactorModel(task="classification", config=small_config())

    model.fit(x.iloc[:120], direction.iloc[:120], x.iloc[120:150], direction.iloc[120:150])
    probabilities = model.predict(x.iloc[150:])
    evaluation = model.evaluate(x.iloc[150:], direction.iloc[150:], realized_returns=y.iloc[150:])

    assert probabilities.between(0, 1).all()
    assert evaluation.accuracy is not None
    assert evaluation.ic > 0.3


def test_signal_strategy_performance_metrics() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    signal = pd.Series([1.0, -1.0, 1.0, -1.0])

    strategy_returns = signal_strategy_returns(signal, returns, transaction_cost_bps=0)

    assert strategy_returns.tolist() == [0.01, 0.02, 0.03, 0.01]
    assert sharpe_ratio(strategy_returns) > 0
    assert max_drawdown(strategy_returns) == 0.0
