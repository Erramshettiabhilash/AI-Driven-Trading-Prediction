"""Combine model prediction columns into an ensemble signal.

Example:
    python scripts/run_ensemble.py ^
        --input results/predictions.csv ^
        --prediction-columns xgboost lstm linear_baseline ^
        --realized-column realized_return ^
        --method ic_weighted ^
        --output results/ensemble/ensemble_predictions.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from models import (
    RidgeStackingEnsemble,
    evaluate_prediction_signals,
    ic_weighted_ensemble,
    regime_conditional_ensemble,
    regime_conditional_weights,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ensemble generation."""
    parser = argparse.ArgumentParser(description="Run model prediction ensembling.")
    parser.add_argument("--input", required=True, help="CSV with timestamp, prediction, and realized columns.")
    parser.add_argument("--prediction-columns", nargs="+", required=True, help="Base model prediction columns.")
    parser.add_argument("--realized-column", default="realized_return", help="Realized return column.")
    parser.add_argument(
        "--method",
        choices=["ic_weighted", "ridge_stacking", "regime_conditional"],
        default="ic_weighted",
        help="Ensemble method.",
    )
    parser.add_argument("--regime-column", default=None, help="Regime column for regime-conditional ensemble.")
    parser.add_argument("--window", type=int, default=20, help="Rolling IC window.")
    parser.add_argument("--output", required=True, help="Output prediction CSV path.")
    parser.add_argument("--metrics-output", default=None, help="Optional metrics CSV path.")
    return parser.parse_args()


def main() -> None:
    """Create ensemble predictions and optional metrics from a CSV."""
    args = parse_args()
    frame = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    predictions = frame[args.prediction_columns]
    realized = frame[args.realized_column]

    if args.method == "ic_weighted":
        ensemble, _ = ic_weighted_ensemble(predictions, realized, window=args.window)
    elif args.method == "ridge_stacking":
        split_at = max(1, int(len(predictions) * 0.7))
        stacker = RidgeStackingEnsemble().fit(predictions.iloc[:split_at], realized.iloc[:split_at])
        ensemble = stacker.predict(predictions)
    else:
        if args.regime_column is None:
            raise ValueError("--regime-column is required for regime_conditional method.")
        weights = regime_conditional_weights(predictions, realized, frame[args.regime_column])
        ensemble = regime_conditional_ensemble(predictions, frame[args.regime_column], weights)

    output = predictions.copy()
    output["ensemble_prediction"] = ensemble
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path)

    if args.metrics_output:
        evaluation = evaluate_prediction_signals(predictions, realized, ensemble)
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation.metrics.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    main()
