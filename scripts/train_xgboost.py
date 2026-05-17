"""Train a Step 5 XGBoost factor model from a feature CSV.

Example:
    python scripts/train_xgboost.py --input data/processed/SPY_features.csv --task regression
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from evaluation import temporal_train_test_split
from models import XGBoostFactorModel, XGBoostModelConfig, build_factor_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for XGBoost training."""
    parser = argparse.ArgumentParser(description="Train an XGBoost factor model.")
    parser.add_argument("--input", required=True, help="Feature CSV with a timestamp column.")
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Model task.",
    )
    parser.add_argument("--target", default=None, help="Optional target column override.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for test.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Fraction of train for validation.")
    parser.add_argument("--purge-bars", type=int, default=5, help="Bars purged before test.")
    parser.add_argument("--embargo-bars", type=int, default=5, help="Bars embargoed before test.")
    parser.add_argument("--output-dir", default="results/xgboost", help="Directory for model and metrics.")
    return parser.parse_args()


def main() -> None:
    """Train, evaluate, and persist an XGBoost factor model."""
    args = parse_args()
    frame = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    target = args.target or ("target_return_1" if args.task == "regression" else "target_direction_1")
    dataset = build_factor_dataset(frame, target_column=target)

    train_idx, test_idx = temporal_train_test_split(
        dataset.x,
        test_size=args.test_size,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
    )
    x_train_full = dataset.x.iloc[train_idx]
    y_train_full = dataset.y.iloc[train_idx]
    test_x = dataset.x.iloc[test_idx]
    test_y = dataset.y.iloc[test_idx]
    test_returns = dataset.realized_returns.iloc[test_idx]

    valid_count = max(1, int(len(x_train_full) * args.validation_size))
    train_x = x_train_full.iloc[:-valid_count]
    train_y = y_train_full.iloc[:-valid_count]
    valid_x = x_train_full.iloc[-valid_count:]
    valid_y = y_train_full.iloc[-valid_count:]

    model = XGBoostFactorModel(task=args.task, config=XGBoostModelConfig())
    model.fit(train_x, train_y, valid_x, valid_y)
    evaluation = model.evaluate(test_x, test_y, realized_returns=test_returns)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"xgboost_{args.task}.json"
    metrics_path = output_dir / f"xgboost_{args.task}_metrics.json"
    features_path = output_dir / f"xgboost_{args.task}_features.json"

    model.save_model(model_path)
    metrics_path.write_text(
        json.dumps(asdict(evaluation), indent=2),
        encoding="utf-8",
    )
    features_path.write_text(json.dumps(list(dataset.x.columns), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": str(model_path),
                "metrics": str(metrics_path),
                "features": str(features_path),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
