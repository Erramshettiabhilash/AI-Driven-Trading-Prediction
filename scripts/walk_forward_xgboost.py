"""Run a walk-forward XGBoost research pipeline.

Example:
    python scripts/walk_forward_xgboost.py --input data/processed/SPY_features_alt.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation import WalkForwardConfig, plot_rolling_ic, retrain_triggers, run_walk_forward_model
from models import XGBoostFactorModel, XGBoostModelConfig, build_factor_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for walk-forward XGBoost research."""
    parser = argparse.ArgumentParser(description="Run walk-forward XGBoost validation.")
    parser.add_argument("--input", required=True, help="Feature CSV with timestamp column.")
    parser.add_argument("--target", default="target_return_1", help="Target column.")
    parser.add_argument("--initial-train-years", type=int, default=2)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--rolling-ic-window", type=int, default=20)
    parser.add_argument("--ic-threshold", type=float, default=0.02)
    parser.add_argument("--output-dir", default="results/walk_forward")
    return parser.parse_args()


def main() -> None:
    """Run walk-forward XGBoost validation and save OOS artifacts."""
    args = parse_args()
    frame = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    dataset = build_factor_dataset(frame, target_column=args.target)
    config = WalkForwardConfig(
        initial_train_years=args.initial_train_years,
        test_months=args.test_months,
    )

    result = run_walk_forward_model(
        dataset.x,
        dataset.y,
        model_factory=lambda: XGBoostFactorModel(
            "regression",
            XGBoostModelConfig(n_jobs=1),
        ),
        config=config,
        realized_returns=dataset.realized_returns,
        rolling_ic_window=args.rolling_ic_window,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "oos_predictions.csv"
    windows_path = output_dir / "windows.csv"
    metrics_path = output_dir / "metrics.json"
    rolling_ic_path = output_dir / "rolling_ic.csv"
    triggers_path = output_dir / "retrain_triggers.csv"
    chart_path = output_dir / "rolling_ic.html"

    result.predictions.to_csv(predictions_path)
    result.windows.to_csv(windows_path, index=False)
    result.rolling_ic.to_csv(rolling_ic_path)
    retrain_triggers(result.rolling_ic, ic_threshold=args.ic_threshold).to_csv(triggers_path)
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    try:
        plot_rolling_ic(result.rolling_ic, threshold=args.ic_threshold).write_html(chart_path)
    except ImportError:
        chart_path = Path("")

    print(
        json.dumps(
            {
                "predictions": str(predictions_path),
                "windows": str(windows_path),
                "metrics": str(metrics_path),
                "rolling_ic": str(rolling_ic_path),
                "retrain_triggers": str(triggers_path),
                "rolling_ic_chart": str(chart_path) if str(chart_path) else None,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
