"""Train a regime-aware XGBoost model from a feature CSV.

Example:
    python scripts/train_regime_xgboost.py --input data/processed/SPY_features.csv --regime-column adx_regime
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from evaluation import temporal_train_test_split
from features import RegimeDetector
from models import RegimeAwareXGBoostModel, XGBoostModelConfig, build_factor_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for regime-aware training."""
    parser = argparse.ArgumentParser(description="Train a regime-aware XGBoost model.")
    parser.add_argument("--input", required=True, help="Feature CSV with timestamp column.")
    parser.add_argument("--target", default="target_return_1", help="Target column.")
    parser.add_argument(
        "--regime-column",
        default="adx_regime",
        help="Regime column to use. Created automatically if missing.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for test.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Fraction of train for validation.")
    parser.add_argument("--min-regime-samples", type=int, default=30, help="Minimum samples per regime.")
    parser.add_argument("--output-dir", default="results/regime_xgboost", help="Artifact directory.")
    return parser.parse_args()


def main() -> None:
    """Train and evaluate a regime-aware XGBoost model."""
    args = parse_args()
    frame = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    if args.regime_column not in frame.columns:
        frame = RegimeDetector().add_all_regimes(frame)

    dataset = build_factor_dataset(frame, target_column=args.target)
    regimes = frame[args.regime_column].reindex(dataset.x.index)
    train_idx, test_idx = temporal_train_test_split(dataset.x, test_size=args.test_size)

    x_train_full = dataset.x.iloc[train_idx]
    y_train_full = dataset.y.iloc[train_idx]
    regimes_train_full = regimes.iloc[train_idx]
    test_x = dataset.x.iloc[test_idx]
    test_y = dataset.y.iloc[test_idx]
    test_regimes = regimes.iloc[test_idx]
    test_returns = dataset.realized_returns.iloc[test_idx]

    valid_count = max(1, int(len(x_train_full) * args.validation_size))
    fit_x = x_train_full.iloc[:-valid_count]
    fit_y = y_train_full.iloc[:-valid_count]
    fit_regimes = regimes_train_full.iloc[:-valid_count]
    valid_x = x_train_full.iloc[-valid_count:]
    valid_y = y_train_full.iloc[-valid_count:]
    valid_regimes = regimes_train_full.iloc[-valid_count:]

    model = RegimeAwareXGBoostModel(
        task="regression",
        model_config=XGBoostModelConfig(),
        min_regime_samples=args.min_regime_samples,
    )
    model.fit(fit_x, fit_y, fit_regimes, valid_x, valid_y, valid_regimes)
    evaluation = model.evaluate(test_x, test_y, test_regimes, realized_returns=test_returns)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "regime_xgboost_metrics.json"
    regimes_path = output_dir / "trained_regimes.json"

    metrics_path.write_text(json.dumps(asdict(evaluation), indent=2), encoding="utf-8")
    regimes_path.write_text(json.dumps(sorted(model.regime_models), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "trained_regimes": str(regimes_path),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
