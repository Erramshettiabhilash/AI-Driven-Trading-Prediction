"""Train a Stable-Baselines3 RL trading agent from a feature CSV.

Example:
    python scripts/train_rl_agent.py --input data/processed/SPY_features.csv --agent PPO
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import pandas as pd

from rl import (
    RLTrainingConfig,
    TradingEnvironment,
    TradingEnvironmentConfig,
    evaluate_agent,
    save_agent,
    train_agent,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RL training."""
    parser = argparse.ArgumentParser(description="Train an RL trading agent.")
    parser.add_argument("--input", required=True, help="Feature CSV with timestamp column.")
    parser.add_argument("--agent", choices=["DQN", "PPO", "A2C"], default="PPO", help="RL agent.")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Training timesteps.")
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0, help="Transaction cost.")
    parser.add_argument("--drawdown-penalty", type=float, default=0.1, help="Reward drawdown penalty.")
    parser.add_argument("--output-dir", default="results/rl", help="Directory for model and metrics.")
    return parser.parse_args()


def main() -> None:
    """Train an RL agent, evaluate one episode, and save artifacts."""
    args = parse_args()
    frame = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    environment = TradingEnvironment(
        frame,
        config=TradingEnvironmentConfig(
            transaction_cost_bps=args.transaction_cost_bps,
            drawdown_penalty=args.drawdown_penalty,
        ),
    )
    training_config = RLTrainingConfig(agent=args.agent, total_timesteps=args.timesteps)
    model = train_agent(environment, training_config)

    evaluation_env = TradingEnvironment(
        frame,
        config=TradingEnvironmentConfig(
            transaction_cost_bps=args.transaction_cost_bps,
            drawdown_penalty=args.drawdown_penalty,
        ),
    )
    returns = evaluate_agent(model, evaluation_env)

    from evaluation import max_drawdown, sharpe_ratio

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_agent(model, output_dir / f"{args.agent.lower()}_trading_agent")
    metrics_path = output_dir / f"{args.agent.lower()}_metrics.json"
    returns_path = output_dir / f"{args.agent.lower()}_returns.csv"

    metrics = {
        "agent": args.agent,
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "observations": int(returns.dropna().shape[0]),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    returns.to_csv(returns_path)

    print(
        json.dumps(
            {
                "model": str(model_path),
                "metrics": str(metrics_path),
                "returns": str(returns_path),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
