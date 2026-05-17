"""Simulate Step 17 order execution against OHLCV bars.

Example:
    python scripts/simulate_execution.py ^
        --bars data/processed/BTCUSDT_5m.csv ^
        --orders results/orders.csv ^
        --output-fills results/execution/fills.csv ^
        --output-positions results/execution/positions.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from live import ExecutionConfig, ExecutionSimulator, Order, OrderSide, OrderType


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for execution simulation."""
    parser = argparse.ArgumentParser(description="Simulate order execution against OHLCV bars.")
    parser.add_argument("--bars", required=True, help="CSV with timestamp, symbol, open, high, low, close, volume.")
    parser.add_argument("--orders", required=True, help="CSV with order instructions.")
    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Adverse slippage in basis points.")
    parser.add_argument("--market-impact-bps", type=float, default=1.0, help="Market impact in basis points.")
    parser.add_argument("--output-fills", default="results/execution/fills.csv", help="Output fills CSV.")
    parser.add_argument("--output-positions", default="results/execution/positions.csv", help="Output positions CSV.")
    return parser.parse_args()


def read_time_indexed_csv(path: str, timestamp_column: str) -> pd.DataFrame:
    """Read a timestamped CSV sorted by time."""
    frame = pd.read_csv(path)
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
    return frame.sort_values(timestamp_column).set_index(timestamp_column)


def row_to_order(row: pd.Series) -> Order:
    """Convert an order CSV row to an ``Order`` dataclass."""
    return Order(
        order_id=str(row["order_id"]),
        symbol=str(row["symbol"]).upper(),
        side=OrderSide(str(row["side"]).upper()),
        order_type=OrderType(str(row["order_type"]).upper()),
        quantity=float(row["quantity"]),
        timestamp=pd.Timestamp(row.name),
        limit_price=None if pd.isna(row.get("limit_price")) else float(row.get("limit_price")),
        stop_price=None if pd.isna(row.get("stop_price")) else float(row.get("stop_price")),
        take_profit_price=None if pd.isna(row.get("take_profit_price")) else float(row.get("take_profit_price")),
        trailing_distance=None if pd.isna(row.get("trailing_distance")) else float(row.get("trailing_distance")),
    )


def main() -> None:
    """Run execution simulation and write fills plus final positions."""
    args = parse_args()
    bars = read_time_indexed_csv(args.bars, args.timestamp_column)
    orders = read_time_indexed_csv(args.orders, args.timestamp_column)
    if "symbol" not in bars.columns:
        raise KeyError("Bars CSV must include a symbol column.")

    simulator = ExecutionSimulator(
        ExecutionConfig(slippage_bps=args.slippage_bps, market_impact_bps=args.market_impact_bps),
    )
    fill_records: list[dict] = []
    submitted: set[str] = set()

    for timestamp, bar in bars.iterrows():
        due_orders = orders.loc[orders.index <= timestamp]
        for _, order_row in due_orders.iterrows():
            order_id = str(order_row["order_id"])
            if order_id in submitted:
                continue
            simulator.submit_order(row_to_order(order_row))
            submitted.add(order_id)

        fills = simulator.process_bar(str(bar["symbol"]), bar)
        for fill in fills:
            fill_records.append(
                {
                    "timestamp": fill.timestamp,
                    "order_id": fill.order_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "slippage_bps": fill.slippage_bps,
                    "notional": fill.notional,
                },
            )

    positions = [
        {
            "symbol": position.symbol,
            "quantity": position.quantity,
            "average_price": position.average_price,
            "last_price": position.last_price,
            "market_value": position.market_value,
        }
        for position in simulator.positions.values()
    ]

    fills_path = Path(args.output_fills)
    positions_path = Path(args.output_positions)
    fills_path.parent.mkdir(parents=True, exist_ok=True)
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fill_records).to_csv(fills_path, index=False)
    pd.DataFrame(positions).to_csv(positions_path, index=False)


if __name__ == "__main__":
    main()
