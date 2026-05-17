import json

import numpy as np
import pandas as pd

from evaluation import (
    adaptive_retraining_report,
    build_monitoring_snapshot,
    monitoring_alert_messages,
    monitoring_metrics,
    psi_drift_alerts,
    read_live_predictions,
    rolling_live_ic,
)


def sample_predictions(length: int = 40) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC", name="timestamp")
    realized = pd.Series(np.linspace(-0.02, 0.02, length), index=index)
    return pd.DataFrame(
        {
            "prediction": realized,
            "realized_return": realized,
            "strategy_return": realized * 0.5,
            "confidence": np.linspace(0.4, 0.9, length),
        },
        index=index,
    )


def test_read_live_predictions_supports_jsonl(tmp_path) -> None:
    path = tmp_path / "signals.jsonl"
    rows = [
        {"timestamp": "2024-01-01T00:00:00Z", "prediction": 0.01, "realized_return": 0.02},
        {"timestamp": "2024-01-02T00:00:00Z", "prediction": -0.01, "realized_return": -0.02},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    frame = read_live_predictions(path)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame["prediction"].iloc[0] == 0.01


def test_rolling_live_ic_and_metrics_are_computed() -> None:
    predictions = sample_predictions()

    rolling_ic = rolling_live_ic(predictions, window=10)
    metrics = monitoring_metrics(predictions, rolling_ic)

    assert rolling_ic.dropna().iloc[-1] > 0.99
    assert metrics["latest_rolling_ic"] > 0.99
    assert metrics["average_confidence"] > 0.0
    assert np.isfinite(metrics["max_drawdown"])


def test_psi_drift_and_retraining_report_trigger_alerts() -> None:
    index = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    reference = pd.DataFrame({"feature_a": np.arange(100), "feature_b": 1.0}, index=index)
    current = pd.DataFrame({"feature_a": np.arange(100, 200), "feature_b": 1.0}, index=index)
    rolling_ic = pd.Series([0.05, 0.01, 0.0], index=index[:3])

    drift = psi_drift_alerts(reference, current, psi_threshold=0.2)
    retrain = adaptive_retraining_report(rolling_ic, drift, ic_threshold=0.02, consecutive_periods=2)
    messages = monitoring_alert_messages({"latest_rolling_ic": 0.0}, drift, retrain)

    assert bool(drift.loc[drift["feature"].eq("feature_a"), "drift_alert"].iloc[0])
    assert bool(retrain["adaptive_retrain_trigger"].iloc[-1])
    assert any("Feature drift" in message for message in messages)


def test_build_monitoring_snapshot_collects_alerts() -> None:
    predictions = sample_predictions()
    predictions["prediction"] = -predictions["realized_return"]
    reference = pd.DataFrame({"feature": np.arange(100)})
    current = pd.DataFrame({"feature": np.arange(100, 200)})

    snapshot = build_monitoring_snapshot(
        predictions,
        reference,
        current,
        rolling_ic_window=10,
        ic_threshold=0.02,
        psi_threshold=0.2,
    )

    assert snapshot.metrics["latest_rolling_ic"] < 0
    assert not snapshot.drift_report.empty
    assert snapshot.alerts
