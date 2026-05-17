"""Backtest Step 15 signal-strength portfolio construction from CSV inputs.

Example:
    python scripts/backtest_portfolio.py ^
        --input results/predictions.csv ^
        --prediction-columns SPY_pred QQQ_pred GLD_pred ^
        --return-columns SPY QQQ GLD ^
        --asset-names SPY QQQ GLD ^
        --benchmark-column SPY ^
        --output-dir results/portfolio
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation import ExecutionCostConfig, backtest_signal_portfolio, prediction_to_signal


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for portfolio backtesting."""
    parser = argparse.ArgumentParser(description="Run execution-aware portfolio backtest.")
    parser.add_argument("--input", required=True, help="CSV with timestamp, predictions, and returns.")
    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--prediction-columns", nargs="+", required=True, help="Prediction score columns.")
    parser.add_argument("--return-columns", nargs="+", required=True, help="Realized return columns.")
    parser.add_argument(
        "--asset-names",
        nargs="*",
        default=None,
        help="Optional asset names used to align prediction and return columns.",
    )
    parser.add_argument("--benchmark-column", default=None, help="Optional benchmark return column.")
    parser.add_argument("--threshold", type=float, default=0.001, help="BUY/SELL score threshold.")
    parser.add_argument("--max-position-weight", type=float, default=0.05, help="Max weight per asset.")
    parser.add_argument("--max-gross-leverage", type=float, default=1.0, help="Gross exposure cap.")
    parser.add_argument("--long-only", action="store_true", help="Disable short weights.")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage in basis points.")
    parser.add_argument("--spread-bps", type=float, default=1.0, help="Bid-ask spread in basis points.")
    parser.add_argument("--commission-bps", type=float, default=0.0, help="Commission in basis points.")
    parser.add_argument("--annualization-factor", type=int, default=252, help="Periods per year.")
    parser.add_argument("--output-dir", default="results/portfolio", help="Directory for report artifacts.")
    return parser.parse_args()


def read_frame(path: str, timestamp_column: str) -> pd.DataFrame:
    """Read a CSV and use timestamps as the index when present."""
    frame = pd.read_csv(path)
    if timestamp_column in frame.columns:
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
        frame = frame.set_index(timestamp_column).sort_index()
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required CSV columns are missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def json_ready(value: Any) -> Any:
    """Convert pandas and NumPy scalars into JSON-compatible values."""
    if hasattr(value, "item"):
        return value.item()
    return value


def aligned_asset_frame(frame: pd.DataFrame, columns: list[str], asset_names: list[str] | None) -> pd.DataFrame:
    """Select columns and optionally rename them to shared asset names."""
    output = frame[columns].astype(float).copy()
    if asset_names is not None:
        if len(asset_names) != len(columns):
            raise ValueError("--asset-names length must match selected columns.")
        output.columns = asset_names
    return output


def main() -> None:
    """Run portfolio construction and write weights, returns, and metrics."""
    args = parse_args()
    frame = read_frame(args.input, args.timestamp_column)
    require_columns(frame, [*args.prediction_columns, *args.return_columns])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = aligned_asset_frame(frame, args.prediction_columns, args.asset_names)
    realized_returns = aligned_asset_frame(frame, args.return_columns, args.asset_names)
    benchmark = frame[args.benchmark_column].astype(float) if args.benchmark_column else None

    result = backtest_signal_portfolio(
        predictions,
        realized_returns,
        threshold=args.threshold,
        max_position_weight=args.max_position_weight,
        max_gross_leverage=args.max_gross_leverage,
        allow_short=not args.long_only,
        cost_config=ExecutionCostConfig(
            slippage_bps=args.slippage_bps,
            spread_bps=args.spread_bps,
            commission_bps=args.commission_bps,
        ),
        benchmark_returns=benchmark,
        annualization_factor=args.annualization_factor,
    )

    report = pd.DataFrame(
        {
            "gross_return": result.gross_returns,
            "execution_cost": result.costs,
            "strategy_return": result.returns,
            "equity_curve": result.equity_curve,
            "turnover": result.turnover,
        },
    )
    report.to_csv(output_dir / "portfolio_returns.csv")
    result.target_weights.to_csv(output_dir / "target_weights.csv")

    first_asset = predictions.columns[0]
    prediction_to_signal(predictions[first_asset], args.threshold).to_frame().to_csv(
        output_dir / f"{first_asset}_signals.csv",
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({key: json_ready(value) for key, value in result.metrics.items()}, file, indent=2)


if __name__ == "__main__":
    main()
