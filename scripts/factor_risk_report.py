"""Generate factor risk, attribution, and covariance reports from return CSVs.

Example:
    python scripts/factor_risk_report.py ^
        --input results/walk_forward/oos_predictions.csv ^
        --strategy-column strategy_return ^
        --benchmark-column SPY ^
        --factor-columns MKT SMB HML UMD ^
        --vix-column VIX ^
        --asset-return-columns SPY QQQ GLD BTC ^
        --output-dir results/factor_risk
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation import (
    beta_to_benchmark,
    covariance_matrix,
    fama_french_exposures,
    jensens_alpha,
    ols_factor_regression,
    return_attribution,
    volatility_factor_correlation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for factor risk reporting."""
    parser = argparse.ArgumentParser(description="Create factor risk model reports.")
    parser.add_argument("--input", required=True, help="CSV containing timestamp and return columns.")
    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--strategy-column", default="strategy_return", help="Strategy return column.")
    parser.add_argument("--benchmark-column", default="SPY", help="Benchmark return column.")
    parser.add_argument(
        "--factor-columns",
        nargs="*",
        default=["MKT", "SMB", "HML", "UMD"],
        help="Factor return columns, such as MKT SMB HML UMD.",
    )
    parser.add_argument("--vix-column", default="VIX", help="VIX level column for volatility sensitivity.")
    parser.add_argument(
        "--asset-return-columns",
        nargs="*",
        default=None,
        help="Columns to include in the annualized covariance matrix.",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Annualized risk-free rate.")
    parser.add_argument("--annualization-factor", type=int, default=252, help="Periods per year.")
    parser.add_argument("--use-vix-levels", action="store_true", help="Correlate with VIX levels, not changes.")
    parser.add_argument("--output-dir", default="results/factor_risk", help="Directory for report artifacts.")
    return parser.parse_args()


def read_return_frame(path: str, timestamp_column: str) -> pd.DataFrame:
    """Read a return CSV and use timestamps as the index when available."""
    frame = pd.read_csv(path)
    if timestamp_column in frame.columns:
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
        frame = frame.set_index(timestamp_column).sort_index()
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when a required report column is missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def json_ready(value: Any) -> Any:
    """Convert pandas and NumPy scalar objects into JSON-compatible values."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Series):
        return {key: json_ready(val) for key, val in value.items()}
    return value


def main() -> None:
    """Build factor risk report artifacts from a CSV of aligned returns."""
    args = parse_args()
    frame = read_return_frame(args.input, args.timestamp_column)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    require_columns(frame, [args.strategy_column, args.benchmark_column])
    strategy_returns = frame[args.strategy_column].astype(float)
    benchmark_returns = frame[args.benchmark_column].astype(float)
    available_factors = [column for column in args.factor_columns if column in frame.columns]

    beta = beta_to_benchmark(strategy_returns, benchmark_returns)
    alpha = jensens_alpha(
        strategy_returns,
        benchmark_returns,
        risk_free_rate=args.risk_free_rate,
        annualization_factor=args.annualization_factor,
    )
    benchmark_regression = ols_factor_regression(
        strategy_returns,
        benchmark_returns.to_frame(args.benchmark_column),
    )

    summary: dict[str, Any] = {
        "strategy_column": args.strategy_column,
        "benchmark_column": args.benchmark_column,
        "beta_to_benchmark": beta,
        "jensens_alpha_annualized": alpha,
        "benchmark_r_squared": benchmark_regression.r_squared,
        "factor_columns_used": available_factors,
    }

    if available_factors:
        factor_returns = frame[available_factors].astype(float)
        factor_regression = ols_factor_regression(strategy_returns, factor_returns)
        exposures = fama_french_exposures(strategy_returns, factor_returns, tuple(available_factors))
        attribution = return_attribution(strategy_returns, factor_returns)

        exposures.to_csv(output_dir / "factor_exposures.csv")
        attribution.attribution_table.to_csv(output_dir / "return_attribution.csv")
        factor_regression.residual_returns.to_frame().to_csv(output_dir / "factor_residuals.csv")

        summary["factor_alpha_daily"] = factor_regression.alpha
        summary["factor_alpha_annualized"] = factor_regression.alpha * args.annualization_factor
        summary["factor_betas"] = factor_regression.betas
        summary["factor_r_squared"] = factor_regression.r_squared
        summary["attributed_total_return"] = attribution.total_return

    if args.vix_column in frame.columns:
        summary["vix_correlation"] = volatility_factor_correlation(
            strategy_returns,
            frame[args.vix_column].astype(float),
            use_changes=not args.use_vix_levels,
        )
        summary["vix_correlation_basis"] = "levels" if args.use_vix_levels else "changes"

    covariance_columns = args.asset_return_columns or [args.strategy_column, args.benchmark_column, *available_factors]
    covariance_columns = [column for column in covariance_columns if column in frame.columns]
    if len(covariance_columns) >= 2:
        covariance = covariance_matrix(
            frame[covariance_columns].astype(float),
            annualization_factor=args.annualization_factor,
        )
        covariance.to_csv(output_dir / "covariance_matrix.csv")
        summary["covariance_columns"] = covariance_columns

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({key: json_ready(value) for key, value in summary.items()}, file, indent=2)


if __name__ == "__main__":
    main()
