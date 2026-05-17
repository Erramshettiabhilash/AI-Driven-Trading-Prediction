"""Run Optuna Bayesian optimization for an LSTM time-series model.

Example:
    python scripts/optimize_lstm.py --input data/processed/SPY_features.csv --trials 100
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from models import build_factor_dataset
from optimization import run_lstm_study, save_study_result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LSTM optimization."""
    parser = argparse.ArgumentParser(description="Optimize LSTM hyperparameters with Optuna.")
    parser.add_argument("--input", required=True, help="Feature CSV with timestamp column.")
    parser.add_argument("--target", default="target_return_1", help="Target column.")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials.")
    parser.add_argument("--n-splits", type=int, default=5, help="TimeSeriesSplit folds.")
    parser.add_argument("--test-size", type=int, default=None, help="Optional validation fold size.")
    parser.add_argument("--sequence-length", type=int, default=60, help="LSTM sequence length.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs per trial.")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URI.")
    parser.add_argument("--output", default="results/optimization/lstm_study.json")
    return parser.parse_args()


def main() -> None:
    """Run the Optuna LSTM study and save the best result."""
    args = parse_args()
    frame = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    dataset = build_factor_dataset(frame, target_column=args.target)
    _, result = run_lstm_study(
        x=dataset.x,
        y=dataset.y,
        n_trials=args.trials,
        storage=args.storage,
        objective_kwargs={
            "n_splits": args.n_splits,
            "test_size": args.test_size,
            "sequence_length": args.sequence_length,
            "epochs": args.epochs,
        },
    )
    output_path = save_study_result(result, Path(args.output))
    print(output_path)


if __name__ == "__main__":
    main()
