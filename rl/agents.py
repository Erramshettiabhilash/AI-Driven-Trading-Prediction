"""Stable-Baselines3 agent helpers for the trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from evaluation.performance import max_drawdown, sharpe_ratio
from rl.trading_env import TradingEnvironment


AgentName = Literal["DQN", "PPO", "A2C"]


@dataclass(frozen=True)
class RLTrainingConfig:
    """Configuration for Stable-Baselines3 agent training."""

    agent: AgentName = "PPO"
    policy: str = "MlpPolicy"
    total_timesteps: int = 10_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    seed: int = 42
    verbose: int = 0


def make_agent(env: TradingEnvironment, config: RLTrainingConfig):
    """Create a DQN, PPO, or A2C agent for the trading environment."""
    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install stable-baselines3 to train RL agents.") from exc

    agent_classes = {
        "DQN": DQN,
        "PPO": PPO,
        "A2C": A2C,
    }
    model_class = agent_classes[config.agent]
    return model_class(
        config.policy,
        env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        seed=config.seed,
        verbose=config.verbose,
    )


def train_agent(env: TradingEnvironment, config: RLTrainingConfig):
    """Train and return a Stable-Baselines3 trading agent."""
    model = make_agent(env, config)
    model.learn(total_timesteps=config.total_timesteps)
    return model


def evaluate_agent(model, env: TradingEnvironment, deterministic: bool = True) -> pd.Series:
    """Run a trained policy through one environment episode and return returns."""
    observation, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, _, terminated, truncated, _ = env.step(int(action))

    return env.returns()


def save_agent(model, path: str | Path) -> Path:
    """Save a Stable-Baselines3 model and return the saved path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return output_path


def compare_rl_to_signal_strategy(
    rl_returns: pd.Series,
    signal_returns: pd.Series,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """Compare RL strategy Sharpe and drawdown against a supervised signal strategy."""
    rows = [
        {
            "strategy": "reinforcement_learning",
            "sharpe": sharpe_ratio(rl_returns, annualization_factor=annualization_factor),
            "max_drawdown": max_drawdown(rl_returns),
            "observations": int(pd.Series(rl_returns).dropna().shape[0]),
        },
        {
            "strategy": "xgboost_signal",
            "sharpe": sharpe_ratio(signal_returns, annualization_factor=annualization_factor),
            "max_drawdown": max_drawdown(signal_returns),
            "observations": int(pd.Series(signal_returns).dropna().shape[0]),
        },
    ]
    return pd.DataFrame(rows)
