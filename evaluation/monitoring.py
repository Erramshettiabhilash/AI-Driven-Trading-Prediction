"""Model monitoring, drift detection, and retraining decision utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.walk_forward import feature_drift_report, retrain_triggers
from evaluation.metrics import rolling_information_coefficient
from evaluation.performance import max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Compact state of live model health."""

    rolling_ic: pd.Series
    drift_report: pd.DataFrame
    retrain_report: pd.DataFrame
    metrics: dict[str, float]
    alerts: list[str]


def read_live_predictions(path: str | Path, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Read CSV or JSONL live prediction artifacts into a timestamp-indexed frame."""
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        frame = pd.read_json(input_path, lines=True)
    else:
        frame = pd.read_csv(input_path)
    if timestamp_column in frame.columns:
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
        frame = frame.set_index(timestamp_column).sort_index()
    return frame


def rolling_live_ic(
    predictions: pd.DataFrame,
    prediction_column: str = "prediction",
    realized_column: str = "realized_return",
    window: int = 20,
) -> pd.Series:
    """Return rolling live IC from prediction and realized return columns."""
    missing = {prediction_column, realized_column}.difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing live prediction columns: {sorted(missing)}")
    return rolling_information_coefficient(
        predictions[prediction_column],
        predictions[realized_column],
        window=window,
    )


def monitoring_metrics(
    live_predictions: pd.DataFrame,
    rolling_ic: pd.Series,
    return_column: str = "strategy_return",
    confidence_column: str = "confidence",
) -> dict[str, float]:
    """Summarize live monitoring metrics for dashboards and alerts."""
    metrics = {
        "latest_rolling_ic": float(rolling_ic.dropna().iloc[-1]) if not rolling_ic.dropna().empty else float("nan"),
        "mean_rolling_ic": float(rolling_ic.mean()) if not rolling_ic.dropna().empty else float("nan"),
        "live_observations": float(len(live_predictions)),
        "average_confidence": float(live_predictions[confidence_column].mean())
        if confidence_column in live_predictions
        else float("nan"),
        "sharpe_ratio": float("nan"),
        "max_drawdown": float("nan"),
    }
    if return_column in live_predictions:
        metrics["sharpe_ratio"] = sharpe_ratio(live_predictions[return_column])
        metrics["max_drawdown"] = max_drawdown(live_predictions[return_column])
    return metrics


def psi_drift_alerts(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    psi_threshold: float = 0.2,
    bins: int = 10,
) -> pd.DataFrame:
    """Return PSI drift report for live feature distributions."""
    return feature_drift_report(
        reference_features,
        current_features,
        bins=bins,
        psi_threshold=psi_threshold,
    )


def adaptive_retraining_report(
    rolling_ic: pd.Series,
    drift_report: pd.DataFrame,
    ic_threshold: float = 0.02,
    psi_threshold: float = 0.2,
    consecutive_periods: int = 2,
) -> pd.DataFrame:
    """Combine IC decay and PSI drift into retraining trigger rows."""
    ic_triggers = retrain_triggers(
        rolling_ic,
        ic_threshold=ic_threshold,
        consecutive_periods=consecutive_periods,
    )
    drift_alert = bool((drift_report.get("psi", pd.Series(dtype=float)) >= psi_threshold).any())
    output = ic_triggers.copy()
    output["drift_alert"] = drift_alert
    output["adaptive_retrain_trigger"] = output["retrain_trigger"] | drift_alert
    return output


def monitoring_alert_messages(
    metrics: dict[str, float],
    drift_report: pd.DataFrame,
    retrain_report: pd.DataFrame,
    ic_threshold: float = 0.02,
) -> list[str]:
    """Create concise alert messages for monitoring breaches."""
    messages: list[str] = []
    latest_ic = metrics.get("latest_rolling_ic", float("nan"))
    if np.isfinite(latest_ic) and latest_ic < ic_threshold:
        messages.append(f"Rolling IC below threshold: {latest_ic:.4f} < {ic_threshold:.4f}")
    if not drift_report.empty and bool(drift_report["drift_alert"].any()):
        features = drift_report.loc[drift_report["drift_alert"], "feature"].head(5).tolist()
        messages.append(f"Feature drift alert on: {', '.join(map(str, features))}")
    if not retrain_report.empty and bool(retrain_report["adaptive_retrain_trigger"].iloc[-1]):
        messages.append("Adaptive retraining trigger is active.")
    return messages


def build_monitoring_snapshot(
    live_predictions: pd.DataFrame,
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    prediction_column: str = "prediction",
    realized_column: str = "realized_return",
    rolling_ic_window: int = 20,
    ic_threshold: float = 0.02,
    psi_threshold: float = 0.2,
    consecutive_periods: int = 2,
) -> MonitoringSnapshot:
    """Build a complete monitoring snapshot from live predictions and features."""
    rolling_ic = rolling_live_ic(
        live_predictions,
        prediction_column=prediction_column,
        realized_column=realized_column,
        window=rolling_ic_window,
    )
    drift = psi_drift_alerts(reference_features, current_features, psi_threshold=psi_threshold)
    retrain = adaptive_retraining_report(
        rolling_ic,
        drift,
        ic_threshold=ic_threshold,
        psi_threshold=psi_threshold,
        consecutive_periods=consecutive_periods,
    )
    metrics = monitoring_metrics(live_predictions, rolling_ic)
    alerts = monitoring_alert_messages(metrics, drift, retrain, ic_threshold=ic_threshold)
    return MonitoringSnapshot(
        rolling_ic=rolling_ic,
        drift_report=drift,
        retrain_report=retrain,
        metrics=metrics,
        alerts=alerts,
    )


def log_monitoring_to_mlflow(
    snapshot: MonitoringSnapshot,
    experiment_name: str = "ai_quant_research_platform",
    tracking_uri: str = "file:./results/mlruns",
    run_name: str = "live_monitoring",
    params: dict[str, Any] | None = None,
) -> None:
    """Log monitoring metrics and alert counts to MLflow."""
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install mlflow to log monitoring runs.") from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics({key: value for key, value in snapshot.metrics.items() if np.isfinite(value)})
        mlflow.log_metric("alert_count", float(len(snapshot.alerts)))
        mlflow.log_metric("drift_feature_count", float(snapshot.drift_report["drift_alert"].sum()))
