import numpy as np
import pandas as pd

from evaluation import (
    WalkForwardConfig,
    feature_drift_report,
    generate_walk_forward_windows,
    population_stability_index,
    retrain_triggers,
    run_walk_forward_model,
)


class TinyLinearModel:
    def __init__(self) -> None:
        self.coef_: float = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series, *_args) -> "TinyLinearModel":
        numerator = float((x["feature"] * y).sum())
        denominator = float((x["feature"] ** 2).sum())
        self.coef_ = numerator / denominator if denominator else 0.0
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(x["feature"] * self.coef_, index=x.index, name="prediction")


def sample_walk_forward_data() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", "2023-12-31", freq="D", tz="UTC", name="timestamp")
    feature = np.sin(np.arange(len(index)) / 20)
    x = pd.DataFrame({"feature": feature}, index=index)
    y = pd.Series(feature * 0.01, index=index, name="target_return_1")
    return x, y


def test_generate_walk_forward_windows_uses_calendar_boundaries() -> None:
    x, _ = sample_walk_forward_data()
    windows = generate_walk_forward_windows(
        x,
        WalkForwardConfig(initial_train_years=2, test_months=3, min_train_observations=10),
    )

    assert len(windows) >= 7
    assert windows[0].train_start == pd.Timestamp("2020-01-01", tz="UTC")
    assert windows[0].train_end == pd.Timestamp("2022-01-01", tz="UTC")
    assert windows[0].test_start == pd.Timestamp("2022-01-01", tz="UTC")
    assert windows[0].test_end == pd.Timestamp("2022-04-01", tz="UTC")


def test_run_walk_forward_model_collects_oos_predictions_only() -> None:
    x, y = sample_walk_forward_data()
    result = run_walk_forward_model(
        x,
        y,
        model_factory=TinyLinearModel,
        config=WalkForwardConfig(
            initial_train_years=2,
            test_months=3,
            validation_fraction=0.1,
            min_train_observations=30,
            min_test_observations=5,
        ),
        rolling_ic_window=20,
    )

    assert result.predictions.index.min() >= pd.Timestamp("2022-01-01", tz="UTC")
    assert {"prediction", "realized_return", "target", "window_id"}.issubset(result.predictions.columns)
    assert result.metrics["oos_ic"] > 0.95
    assert not result.windows.empty


def test_population_stability_index_detects_distribution_shift() -> None:
    stable = population_stability_index(pd.Series(range(100)), pd.Series(range(100)))
    shifted = population_stability_index(pd.Series(range(100)), pd.Series(range(100, 200)))

    assert stable < 0.01
    assert shifted > stable


def test_feature_drift_report_flags_large_psi() -> None:
    train = pd.DataFrame({"feature_a": range(100), "feature_b": [1.0] * 100})
    test = pd.DataFrame({"feature_a": range(100, 200), "feature_b": [1.0] * 100})

    report = feature_drift_report(train, test, psi_threshold=0.2)

    feature_a = report.loc[report["feature"].eq("feature_a")].iloc[0]
    assert feature_a["drift_alert"]
    assert feature_a["drift_level"] == "drift"


def test_retrain_triggers_when_rolling_ic_decays() -> None:
    rolling_ic = pd.Series(
        [0.05, 0.03, 0.01, 0.0],
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
    )

    triggers = retrain_triggers(rolling_ic, ic_threshold=0.02, consecutive_periods=2)

    assert not bool(triggers["retrain_trigger"].iloc[2])
    assert bool(triggers["retrain_trigger"].iloc[3])
