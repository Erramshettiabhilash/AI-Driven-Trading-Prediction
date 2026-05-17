"""Train a Step 6 LSTM return forecasting model from a feature CSV.

Example:
    python scripts/train_lstm.py --input data/processed/SPY_features.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import temporal_train_test_split
from models import LSTMModelConfig, LSTMTimeSeriesModel, build_factor_dataset, create_lstm_sequences
from models.lstm_timeseries import subset_sequence_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LSTM training."""
    parser = argparse.ArgumentParser(description="Train an LSTM return forecasting model.")
    parser.add_argument("--input", required=True, help="Feature CSV with a timestamp column.")
    parser.add_argument("--target", default="target_return_1", help="Regression target column.")
    parser.add_argument("--sequence-length", type=int, default=60, help="Number of timesteps per sample.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for test.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Fraction of train for validation.")
    parser.add_argument("--purge-bars", type=int, default=5, help="Bars purged before test.")
    parser.add_argument("--embargo-bars", type=int, default=5, help="Bars embargoed before test.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--output-dir", default="results/lstm", help="Directory for model and metrics.")
    return parser.parse_args()


def main() -> None:
    """Train, evaluate, and persist an LSTM model."""
    args = parse_args()
    frame = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    factor_dataset = build_factor_dataset(frame, target_column=args.target)
    sequence_dataset = create_lstm_sequences(
        factor_dataset.x,
        factor_dataset.y,
        sequence_length=args.sequence_length,
    )

    train_idx, test_idx = temporal_train_test_split(
        pd.Series(index=sequence_dataset.index, dtype=float),
        test_size=args.test_size,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
    )
    valid_count = max(1, int(len(train_idx) * args.validation_size))
    fit_idx = train_idx[:-valid_count]
    valid_idx = train_idx[-valid_count:]

    train = subset_sequence_dataset(sequence_dataset, fit_idx)
    validation = subset_sequence_dataset(sequence_dataset, valid_idx)
    test = subset_sequence_dataset(sequence_dataset, test_idx)

    config = LSTMModelConfig(
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model = LSTMTimeSeriesModel(config=config)
    model.fit(train, validation=validation)

    realized_returns = factor_dataset.realized_returns.reindex(sequence_dataset.index)
    evaluation = model.evaluate(test, realized_returns=realized_returns)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lstm_return_forecaster.keras"
    metrics_path = output_dir / "lstm_metrics.json"
    features_path = output_dir / "lstm_features.json"

    model.save_model(model_path)
    metrics_path.write_text(json.dumps(asdict(evaluation), indent=2), encoding="utf-8")
    features_path.write_text(json.dumps(sequence_dataset.feature_names, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": str(model_path),
                "metrics": str(metrics_path),
                "features": str(features_path),
                "train_sequences": int(len(train.y)),
                "validation_sequences": int(len(validation.y)),
                "test_sequences": int(len(test.y)),
                "target_mean": float(np.mean(train.y)),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
