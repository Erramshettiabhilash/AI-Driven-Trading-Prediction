"""Generate deterministic demo artifacts for the Streamlit dashboard.

This is a local smoke-test utility. It does not train a real model or claim
live trading performance; it creates plausible-looking outputs so the Step 19
dashboard can be viewed before the full data/model pipeline has produced files.

Example:
    python scripts/generate_dashboard_demo_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Write demo signal, portfolio, and SHAP artifacts."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2026-04-01", periods=80, freq="D", tz="UTC", name="timestamp")

    realized = pd.Series(rng.normal(0.0006, 0.012, len(index)), index=index)
    prediction = realized.rolling(3, min_periods=1).mean() + rng.normal(0.0, 0.002, len(index))
    confidence = prediction.abs().clip(upper=0.02) / 0.02
    signal = pd.Series(
        np.select([prediction > 0.001, prediction < -0.001], ["BUY", "SELL"], default="HOLD"),
        index=index,
    )
    strategy_return = pd.Series(np.sign(prediction) * realized * 0.6 - 0.00005, index=index)

    live_dir = Path("data/live")
    portfolio_dir = Path("results/portfolio")
    shap_dir = Path("results/explainability")
    live_dir.mkdir(parents=True, exist_ok=True)
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)

    signal_path = live_dir / "signals.jsonl"
    with signal_path.open("w", encoding="utf-8") as file:
        for timestamp in index:
            row = {
                "timestamp": timestamp.isoformat(),
                "symbol": "BTCUSDT",
                "signal": str(signal.loc[timestamp]),
                "confidence": float(confidence.loc[timestamp]),
                "prediction": float(prediction.loc[timestamp]),
                "realized_return": float(realized.loc[timestamp]),
                "strategy_return": float(strategy_return.loc[timestamp]),
                "regime": "trending" if timestamp.day % 3 else "ranging",
            }
            file.write(json.dumps(row) + "\n")

    portfolio = pd.DataFrame(
        {
            "timestamp": index,
            "gross_return": strategy_return + 0.00005,
            "execution_cost": 0.00005,
            "strategy_return": strategy_return,
            "equity_curve": (1 + strategy_return).cumprod(),
            "turnover": rng.uniform(0.01, 0.18, len(index)),
        },
    )
    portfolio.to_csv(portfolio_dir / "portfolio_returns.csv", index=False)

    shap = pd.DataFrame(
        {
            "feature": ["ema_cross", "rsi_14", "volume_delta", "vwap_deviation", "atr_14", "macd_histogram"],
            "mean_abs_shap": [0.042, 0.036, 0.029, 0.022, 0.018, 0.014],
            "importance_rank": [1, 2, 3, 4, 5, 6],
        },
    )
    shap.to_csv(shap_dir / "global_importance.csv", index=False)

    print(f"Wrote {signal_path}")
    print(f"Wrote {portfolio_dir / 'portfolio_returns.csv'}")
    print(f"Wrote {shap_dir / 'global_importance.csv'}")


if __name__ == "__main__":
    main()
