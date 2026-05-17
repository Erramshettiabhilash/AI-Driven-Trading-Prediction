"""Streamlit dashboard and Plotly chart helpers for Step 19 monitoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.monitoring import read_live_predictions
from evaluation.performance import max_drawdown


def equity_curve_from_returns(returns: pd.Series) -> pd.Series:
    """Return cumulative equity curve from simple returns."""
    return (1 + returns.fillna(0.0)).cumprod().rename("equity_curve")


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return drawdown series from an equity curve."""
    return (equity_curve / equity_curve.cummax() - 1).rename("drawdown")


def equity_curve_figure(equity_curve: pd.Series):
    """Create a Plotly equity curve figure."""
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode="lines", name="Equity"))
    figure.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Equity", template="plotly_white")
    return figure


def drawdown_figure(drawdown: pd.Series):
    """Create a Plotly drawdown figure."""
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", fill="tozeroy", name="Drawdown"))
    figure.update_layout(title="Drawdown", xaxis_title="Time", yaxis_title="Drawdown", template="plotly_white")
    return figure


def rolling_ic_figure(rolling_ic: pd.Series, threshold: float = 0.02):
    """Create a Plotly rolling IC figure."""
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=rolling_ic.index, y=rolling_ic, mode="lines", name="Rolling IC"))
    figure.add_hline(y=threshold, line_dash="dash", line_color="red")
    figure.update_layout(title="Rolling IC", xaxis_title="Time", yaxis_title="IC", template="plotly_white")
    return figure


def shap_top_features(shap_importance: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top SHAP features for dashboard display."""
    if shap_importance.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    return shap_importance.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def shap_top_features_figure(shap_importance: pd.DataFrame, top_n: int = 5):
    """Create a horizontal bar chart for top SHAP features."""
    import plotly.express as px

    top = shap_top_features(shap_importance, top_n=top_n)
    return px.bar(
        top.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Top SHAP Features",
        template="plotly_white",
    )


def load_optional_csv(path: str | Path | None) -> pd.DataFrame:
    """Load a CSV if a path exists, otherwise return an empty frame."""
    if path is None:
        return pd.DataFrame()
    input_path = Path(path)
    if not input_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(input_path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp").sort_index()
    return frame


def file_status(path: str | Path) -> tuple[bool, str]:
    """Return whether a dashboard input file exists and a short status message."""
    input_path = Path(path)
    if not input_path.exists():
        return False, f"Missing: {input_path}"
    if input_path.stat().st_size == 0:
        return False, f"Empty file: {input_path}"
    return True, f"Loaded: {input_path}"


def run_streamlit_dashboard() -> None:
    """Run the Streamlit dashboard app."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install streamlit to run the dashboard.") from exc

    from evaluation.monitoring import rolling_live_ic

    st.set_page_config(page_title="AI Quant Monitoring", layout="wide")
    st.title("AI Quant Trading Monitor")

    signals_path = st.sidebar.text_input("Live predictions CSV/JSONL", "data/live/signals.jsonl")
    returns_path = st.sidebar.text_input("Portfolio returns CSV", "results/portfolio/portfolio_returns.csv")
    shap_path = st.sidebar.text_input("SHAP importance CSV", "results/explainability/global_importance.csv")
    ic_threshold = st.sidebar.number_input("Retrain IC threshold", value=0.02, step=0.01, format="%.4f")

    signal_ready, signal_status = file_status(signals_path)
    returns_ready, returns_status = file_status(returns_path)
    shap_ready, shap_status = file_status(shap_path)
    st.sidebar.caption(signal_status)
    st.sidebar.caption(returns_status)
    st.sidebar.caption(shap_status)

    if not signal_ready and not returns_ready and not shap_ready:
        st.info(
            "No dashboard artifacts found yet. Generate demo data with "
            "`python scripts/generate_dashboard_demo_data.py`, or run the training/live pipeline.",
        )

    live_predictions = read_live_predictions(signals_path) if signal_ready else pd.DataFrame()
    portfolio_returns = load_optional_csv(returns_path)
    shap_importance = load_optional_csv(shap_path)

    metric_cols = st.columns(4)
    latest_signal = live_predictions["signal"].iloc[-1] if "signal" in live_predictions and not live_predictions.empty else "N/A"
    latest_confidence = (
        float(live_predictions["confidence"].iloc[-1])
        if "confidence" in live_predictions and not live_predictions.empty
        else float("nan")
    )
    metric_cols[0].metric("Latest Signal", latest_signal)
    metric_cols[1].metric("Confidence", f"{latest_confidence:.2%}" if pd.notna(latest_confidence) else "N/A")

    if "strategy_return" in portfolio_returns:
        equity = equity_curve_from_returns(portfolio_returns["strategy_return"])
        drawdown = drawdown_series(equity)
        metric_cols[2].metric("Max Drawdown", f"{max_drawdown(portfolio_returns['strategy_return']):.2%}")
        metric_cols[3].metric("Equity", f"{equity.iloc[-1]:.3f}")
        st.plotly_chart(equity_curve_figure(equity), use_container_width=True)
        st.plotly_chart(drawdown_figure(drawdown), use_container_width=True)
    elif returns_ready:
        st.warning("Portfolio returns file exists, but it needs a `strategy_return` column.")

    if {"prediction", "realized_return"}.issubset(live_predictions.columns):
        rolling_ic = rolling_live_ic(live_predictions)
        st.plotly_chart(rolling_ic_figure(rolling_ic, threshold=ic_threshold), use_container_width=True)
    elif signal_ready:
        st.warning("Live predictions file exists, but rolling IC needs `prediction` and `realized_return` columns.")

    if not shap_importance.empty and {"feature", "mean_abs_shap"}.issubset(shap_importance.columns):
        st.plotly_chart(shap_top_features_figure(shap_importance), use_container_width=True)
    elif shap_ready:
        st.warning("SHAP file exists, but it needs `feature` and `mean_abs_shap` columns.")


if __name__ == "__main__":
    run_streamlit_dashboard()
