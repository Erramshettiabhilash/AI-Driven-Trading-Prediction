import importlib.util

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None and importlib.util.find_spec("gym") is None,
    reason="gymnasium or gym not installed",
)

from evaluation.performance import signal_strategy_returns
from rl import TradingEnvironment, TradingEnvironmentConfig, compare_rl_to_signal_strategy


def sample_env_data(length: int = 40) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC", name="timestamp")
    close = pd.Series(np.linspace(100, 120, length), index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000,
            "rsi_14": 55.0,
            "ema_cross": 0.5,
            "macd": 0.2,
            "atr_14": 1.0,
        },
        index=index,
    )


def test_environment_reset_returns_expected_observation_shape() -> None:
    env = TradingEnvironment(sample_env_data())

    observation, info = env.reset()

    assert observation.shape == (11,)
    assert info["portfolio_value"] == 100_000.0
    assert env.action_space.n == 3


def test_buy_action_generates_positive_return_in_uptrend() -> None:
    env = TradingEnvironment(
        sample_env_data(),
        config=TradingEnvironmentConfig(transaction_cost_bps=0.0),
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step(1)

    assert reward > 0
    assert not terminated
    assert not truncated
    assert info["step_return"] > 0
    assert env.position == 1.0


def test_sell_action_can_target_short_position() -> None:
    env = TradingEnvironment(
        sample_env_data(),
        config=TradingEnvironmentConfig(transaction_cost_bps=0.0, allow_short=True),
    )
    env.reset()

    env.step(2)

    assert env.position == -1.0


def test_environment_episode_returns_have_expected_length() -> None:
    env = TradingEnvironment(
        sample_env_data(length=8),
        config=TradingEnvironmentConfig(transaction_cost_bps=0.0),
    )
    env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(1)

    assert len(env.returns()) == 7
    assert len(env.equity_curve()) == 8


def test_compare_rl_to_signal_strategy_returns_metric_table() -> None:
    rl_returns = pd.Series([0.01, -0.005, 0.02])
    xgb_signal_returns = signal_strategy_returns(
        signal=pd.Series([1.0, -1.0, 1.0]),
        realized_returns=pd.Series([0.01, 0.005, 0.02]),
    )

    comparison = compare_rl_to_signal_strategy(rl_returns, xgb_signal_returns)

    assert comparison["strategy"].tolist() == ["reinforcement_learning", "xgboost_signal"]
    assert {"sharpe", "max_drawdown", "observations"}.issubset(comparison.columns)
