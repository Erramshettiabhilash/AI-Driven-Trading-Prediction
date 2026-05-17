"""Build Step 3 technical features for processed OHLCV CSV files.

Example:
    python scripts/build_features.py --input data/processed/SPY.csv --output data/processed/SPY_features.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from features import FeatureEngineer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature generation."""
    parser = argparse.ArgumentParser(description="Build technical features for one OHLCV CSV.")
    parser.add_argument("--input", required=True, help="Input processed OHLCV CSV path.")
    parser.add_argument("--output", required=True, help="Output feature CSV path.")
    return parser.parse_args()


def main() -> None:
    """Load a processed OHLCV CSV, add features, and save the result."""
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    frame = pd.read_csv(input_path, parse_dates=["timestamp"], index_col="timestamp")
    features = FeatureEngineer().build_all_features(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path)


if __name__ == "__main__":
    main()
