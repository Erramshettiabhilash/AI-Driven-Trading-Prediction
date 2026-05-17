"""Gym-compatible reinforcement learning trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - import path depends on installed RL stack
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - import path depends on installed RL stack
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None


DEFAULT_STATE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi_14",
    "ema_cross",
    "macd",
    "atr_14",
]


@dataclass(frozen=True)
class TradingEnvironmentConfig:
    """Configuration for the RL trading environment."""

    initial_cash: float = 100_000.0
    transaction_cost_bps: float = 2.0
    rolling_vol_window: int = 20
    drawdown_penalty: float = 0.1
    reward_scaling: float = 1.0
    allow_short: bool = True
    max_episode_steps: int | None = None


class TradingEnvironment(gym.Env if gym is not None else object):
    """Single-asset trading environment with discrete Hold/Buy/Sell actions.

    State includes market features plus portfolio state:

    ``[OHLCV, RSI, EMA, MACD, ATR, portfolio_value_ratio, position]``

    Actions:

    - ``0``: Hold current position
    - ``1``: Buy / target long
    - ``2``: Sell / target short if allowed, otherwise flat
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str] | None = None,
        config: TradingEnvironmentConfig | None = None,
    ) -> None:
        """Create a trading environment from feature-enriched OHLCV data."""
        if spaces is None:
            raise ImportError("Install gymnasium or gym to use TradingEnvironment.")
        if "close" not in data:
            raise KeyError("TradingEnvironment requires a close column.")
        if len(data) < 3:
            raise ValueError("TradingEnvironment requires at least three rows.")

        self.config = config or TradingEnvironmentConfig()
        self.data = data.copy().sort_index()
        self.feature_columns = feature_columns or [
            column for column in DEFAULT_STATE_COLUMNS if column in self.data.columns
        ]
        if not self.feature_columns:
            raise ValueError("No feature columns available for environment state.")

        self.data[self.feature_columns] = self.data[self.feature_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        self.data[self.feature_columns] = self.data[self.feature_columns].ffill().fillna(0.0)
        self.next_returns = self.data["close"].pct_change().shift(-1).fillna(0.0)
        self.realized_volatility = (
            self.data["close"]
            .pct_change()
            .rolling(self.config.rolling_vol_window, min_periods=2)
            .std(ddof=0)
            .fillna(0.0)
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns) + 2,),
            dtype=np.float32,
        )
        self.current_step = 0
        self.portfolio_value = self.config.initial_cash
        self.peak_portfolio_value = self.config.initial_cash
        self.position = 0.0
        self.returns_history: list[float] = []
        self.equity_history: list[float] = []
        self.action_history: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return the first observation."""
        if gym is not None and hasattr(super(), "reset"):
            super().reset(seed=seed)

        self.current_step = 0
        self.portfolio_value = self.config.initial_cash
        self.peak_portfolio_value = self.config.initial_cash
        self.position = 0.0
        self.returns_history = []
        self.equity_history = [self.portfolio_value]
        self.action_history = []
        return self._observation(), self._info(0.0, 0.0, 0.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action and advance one bar."""
        if action not in {0, 1, 2}:
            raise ValueError("Action must be 0=Hold, 1=Buy, or 2=Sell.")

        previous_position = self.position
        target_position = self._target_position(action)
        turnover = abs(target_position - previous_position)
        transaction_cost = turnover * self.config.transaction_cost_bps / 10_000

        market_return = float(self.next_returns.iloc[self.current_step])
        step_return = target_position * market_return - transaction_cost
        self.portfolio_value *= 1.0 + step_return
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        drawdown = self.portfolio_value / self.peak_portfolio_value - 1.0

        volatility = float(self.realized_volatility.iloc[self.current_step])
        risk_adjusted_return = step_return / max(volatility, 1e-8)
        reward = (
            risk_adjusted_return
            - self.config.drawdown_penalty * abs(min(drawdown, 0.0))
            - transaction_cost
        ) * self.config.reward_scaling

        self.position = target_position
        self.returns_history.append(step_return)
        self.equity_history.append(self.portfolio_value)
        self.action_history.append(action)

        self.current_step += 1
        reached_data_end = self.current_step >= len(self.data) - 1
        reached_step_limit = (
            self.config.max_episode_steps is not None
            and len(self.returns_history) >= self.config.max_episode_steps
        )
        terminated = bool(reached_data_end or self.portfolio_value <= 0)
        truncated = bool(reached_step_limit and not terminated)

        return self._observation(), float(reward), terminated, truncated, self._info(
            step_return=step_return,
            transaction_cost=transaction_cost,
            drawdown=drawdown,
        )

    def render(self) -> None:
        """Print a compact human-readable environment state."""
        print(
            f"step={self.current_step} "
            f"portfolio={self.portfolio_value:.2f} "
            f"position={self.position:.0f}"
        )

    def returns(self) -> pd.Series:
        """Return strategy returns generated during the current episode."""
        index = self.data.index[: len(self.returns_history)]
        return pd.Series(self.returns_history, index=index, name="rl_strategy_return")

    def equity_curve(self) -> pd.Series:
        """Return portfolio value history for the current episode."""
        index = self.data.index[: len(self.equity_history)]
        return pd.Series(self.equity_history, index=index, name="portfolio_value")

    def _observation(self) -> np.ndarray:
        """Return current market and portfolio state."""
        safe_step = min(self.current_step, len(self.data) - 1)
        market_state = self.data.iloc[safe_step][self.feature_columns].to_numpy(dtype=np.float32)
        portfolio_state = np.array(
            [
                self.portfolio_value / self.config.initial_cash,
                self.position,
            ],
            dtype=np.float32,
        )
        return np.concatenate([market_state, portfolio_state]).astype(np.float32)

    def _target_position(self, action: int) -> float:
        """Map action integer to target portfolio position."""
        if action == 0:
            return self.position
        if action == 1:
            return 1.0
        return -1.0 if self.config.allow_short else 0.0

    def _info(self, step_return: float, transaction_cost: float, drawdown: float) -> dict[str, Any]:
        """Return diagnostic information for one environment step."""
        return {
            "timestamp": self.data.index[min(self.current_step, len(self.data) - 1)],
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "step_return": step_return,
            "transaction_cost": transaction_cost,
            "drawdown": drawdown,
        }
