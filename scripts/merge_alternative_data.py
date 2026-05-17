"""Merge alternative data CSV files into a market feature CSV.

Example:
    python scripts/merge_alternative_data.py ^
        --market data/processed/SPY_features.csv ^
        --alternative data/external/news_sentiment.csv data/external/macro.csv ^
        --output data/processed/SPY_features_alt.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from data import AlternativeDataMerger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for alternative-data merging."""
    parser = argparse.ArgumentParser(description="Merge alternative data into market features.")
    parser.add_argument("--market", required=True, help="Market feature CSV with timestamp column.")
    parser.add_argument(
        "--alternative",
        nargs="+",
        required=True,
        help="Alternative feature CSV files with timestamp columns.",
    )
    parser.add_argument("--output", required=True, help="Output merged CSV path.")
    parser.add_argument("--tolerance", default=None, help="Optional merge tolerance, e.g. 7D.")
    parser.add_argument("--sentiment-window", type=int, default=3, help="Rolling sentiment momentum window.")
    return parser.parse_args()


def main() -> None:
    """Load CSVs, merge point-in-time alternative data, and save output."""
    args = parse_args()
    market = pd.read_csv(args.market, parse_dates=["timestamp"])
    alternatives = [
        pd.read_csv(path, parse_dates=["timestamp"])
        for path in args.alternative
    ]

    merger = AlternativeDataMerger(rolling_sentiment_window=args.sentiment_window)
    merged = merger.merge_many(market, alternatives, tolerance=args.tolerance)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path)


if __name__ == "__main__":
    main()
