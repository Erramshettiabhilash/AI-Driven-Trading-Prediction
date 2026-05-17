"""Generate Step 19 live monitoring reports from prediction and feature CSVs.

Example:
    python scripts/monitor_live_model.py ^
        --live-predictions data/live/predictions.csv ^
        --reference-features data/processed/train_features.csv ^
        --current-features data/live/current_features.csv ^
        --output-dir results/monitoring
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation import build_monitoring_snapshot, log_monitoring_to_mlflow, read_live_predictions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for live model monitoring."""
    parser = argparse.ArgumentParser(description="Create live monitoring reports.")
    parser.add_argument("--live-predictions", required=True, help="CSV/JSONL with prediction and realized_return.")
    parser.add_argument("--reference-features", required=True, help="Training/reference feature CSV.")
    parser.add_argument("--current-features", required=True, help="Current/live feature CSV.")
    parser.add_argument("--prediction-column", default="prediction", help="Prediction column.")
    parser.add_argument("--realized-column", default="realized_return", help="Realized return column.")
    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column.")
    parser.add_argument("--rolling-ic-window", type=int, default=20, help="Rolling IC window.")
    parser.add_argument("--ic-threshold", type=float, default=0.02, help="Retraining IC threshold.")
    parser.add_argument("--psi-threshold", type=float, default=0.2, help="PSI drift threshold.")
    parser.add_argument("--output-dir", default="results/monitoring", help="Output directory.")
    parser.add_argument("--log-mlflow", action="store_true", help="Log metrics to MLflow.")
    parser.add_argument("--tracking-uri", default="file:./results/mlruns", help="MLflow tracking URI.")
    parser.add_argument("--experiment-name", default="ai_quant_research_platform", help="MLflow experiment.")
    return parser.parse_args()


def read_feature_csv(path: str, timestamp_column: str) -> pd.DataFrame:
    """Read a feature CSV with optional timestamp index."""
    frame = pd.read_csv(path)
    if timestamp_column in frame.columns:
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
        frame = frame.set_index(timestamp_column).sort_index()
    return frame


def json_ready(value: Any) -> Any:
    """Convert pandas and NumPy scalars to JSON-compatible values."""
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    """Build and save monitoring reports."""
    args = parse_args()
    predictions = read_live_predictions(args.live_predictions, timestamp_column=args.timestamp_column)
    reference_features = read_feature_csv(args.reference_features, args.timestamp_column)
    current_features = read_feature_csv(args.current_features, args.timestamp_column)

    snapshot = build_monitoring_snapshot(
        predictions,
        reference_features,
        current_features,
        prediction_column=args.prediction_column,
        realized_column=args.realized_column,
        rolling_ic_window=args.rolling_ic_window,
        ic_threshold=args.ic_threshold,
        psi_threshold=args.psi_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.rolling_ic.to_frame("rolling_ic").to_csv(output_dir / "rolling_ic.csv")
    snapshot.drift_report.to_csv(output_dir / "drift_report.csv", index=False)
    snapshot.retrain_report.to_csv(output_dir / "retrain_report.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({key: json_ready(value) for key, value in snapshot.metrics.items()}, file, indent=2)
    with (output_dir / "alerts.txt").open("w", encoding="utf-8") as file:
        file.write("\n".join(snapshot.alerts))

    if args.log_mlflow:
        log_monitoring_to_mlflow(
            snapshot,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            params={
                "rolling_ic_window": args.rolling_ic_window,
                "ic_threshold": args.ic_threshold,
                "psi_threshold": args.psi_threshold,
            },
        )


if __name__ == "__main__":
    main()
