"""Build Step 18 microstructure features from CSV inputs.

Example:
    python scripts/build_microstructure_features.py ^
        --ohlcv data/processed/BTCUSDT_5m.csv ^
        --trades data/live/BTCUSDT_trades.csv ^
        --order-book data/live/BTCUSDT_book.csv ^
        --output data/processed/BTCUSDT_microstructure.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from features import build_microstructure_features


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for microstructure feature generation."""
    parser = argparse.ArgumentParser(description="Build order-flow and microstructure features.")
    parser.add_argument("--ohlcv", required=True, help="OHLCV CSV with timestamp, open, high, low, close, volume.")
    parser.add_argument("--trades", default=None, help="Optional trades CSV with price, bid, ask, quantity.")
    parser.add_argument("--order-book", default=None, help="Optional order book CSV with bid_qty/ask_qty columns.")
    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--atr-column", default="atr_14", help="ATR column used for VWAP deviation.")
    parser.add_argument("--lookback", type=int, default=20, help="Lookback for sweeps and VWAP bands.")
    parser.add_argument("--volume-multiplier", type=float, default=1.5, help="Volume spike multiplier.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def read_frame(path: str | None, timestamp_column: str) -> pd.DataFrame | None:
    """Read an optional timestamp-indexed CSV."""
    if path is None:
        return None
    frame = pd.read_csv(path)
    if timestamp_column in frame.columns:
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
        frame = frame.set_index(timestamp_column).sort_index()
    return frame


def main() -> None:
    """Create and save microstructure feature CSV."""
    args = parse_args()
    ohlcv = read_frame(args.ohlcv, args.timestamp_column)
    if ohlcv is None:
        raise ValueError("--ohlcv is required.")
    trades = read_frame(args.trades, args.timestamp_column)
    order_book = read_frame(args.order_book, args.timestamp_column)

    features = build_microstructure_features(
        ohlcv,
        trades=trades,
        order_book=order_book,
        atr_column=args.atr_column,
        lookback=args.lookback,
        volume_multiplier=args.volume_multiplier,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path)


if __name__ == "__main__":
    main()
