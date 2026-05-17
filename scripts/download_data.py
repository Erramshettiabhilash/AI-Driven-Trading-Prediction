"""Download and preprocess sample market data from the command line.

Example:
    python scripts/download_data.py --symbols SPY AAPL BTC-USD --start 2020-01-01
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from data.market_data import YahooFinanceCollector, save_market_data
from data.preprocessing import DataPreprocessor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for historical yfinance downloads."""
    parser = argparse.ArgumentParser(description="Download historical OHLCV data.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Yahoo Finance symbols.")
    parser.add_argument("--start", required=True, help="Start date, e.g. 2020-01-01.")
    parser.add_argument("--end", default=None, help="Optional end date.")
    parser.add_argument("--interval", default="1d", help="Bar interval, e.g. 1d, 1h, 15m.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for raw CSV files.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory for processed CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Download raw yfinance data, preprocess it, and save both layers."""
    args = parse_args()
    collector = YahooFinanceCollector()
    preprocessor = DataPreprocessor()

    raw_frames = collector.download_many(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )
    save_market_data(raw_frames, Path(args.raw_dir))

    intraday_frequency = None if args.interval.endswith("d") else args.interval
    processed_frames = {
        symbol: preprocessor.clean_ohlcv(frame, intraday_frequency=intraday_frequency)
        for symbol, frame in raw_frames.items()
    }
    save_market_data(processed_frames, Path(args.processed_dir))


if __name__ == "__main__":
    main()
