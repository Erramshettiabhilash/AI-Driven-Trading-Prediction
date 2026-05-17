import numpy as np
import pandas as pd

from models import (
    LinearRegressionBaseline,
    RidgeStackingEnsemble,
    evaluate_prediction_signals,
    ic_weighted_ensemble,
    regime_conditional_ensemble,
    regime_conditional_weights,
    rolling_model_ic,
)


def sample_predictions(length: int = 80) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC", name="timestamp")
    realized = pd.Series(np.sin(np.arange(length) / 4) * 0.01, index=index, name="realized_return")
    predictions = pd.DataFrame(
        {
            "xgboost": realized + 0.001,
            "lstm": realized * 0.7,
            "linear_baseline": -realized * 0.2,
        },
        index=index,
    )
    regimes = pd.Series(["trend"] * (length // 2) + ["range"] * (length - length // 2), index=index)
    return predictions, realized, regimes


def test_rolling_model_ic_uses_prior_window_only() -> None:
    predictions, realized, _ = sample_predictions(length=30)

    ic = rolling_model_ic(predictions, realized, window=10)

    assert ic.iloc[:10].isna().all().all()
    assert ic["xgboost"].iloc[10] > 0.9
    assert ic["linear_baseline"].iloc[10] < -0.9


def test_ic_weighted_ensemble_returns_prediction_and_weights() -> None:
    predictions, realized, _ = sample_predictions()

    ensemble, weights = ic_weighted_ensemble(predictions, realized, window=20)

    assert ensemble.index.equals(predictions.index)
    assert weights.index.equals(predictions.index)
    assert np.isclose(weights.iloc[-1].abs().sum(), 1.0)
    assert ensemble.notna().all()


def test_ridge_stacking_ensemble_fits_on_model_predictions() -> None:
    predictions, realized, _ = sample_predictions()
    stacker = RidgeStackingEnsemble(alpha=0.1).fit(predictions.iloc[:60], realized.iloc[:60])

    stacked = stacker.predict(predictions.iloc[60:])

    assert len(stacked) == 20
    assert stacked.corr(realized.iloc[60:]) > 0.9


def test_regime_conditional_weights_and_ensemble() -> None:
    predictions, realized, regimes = sample_predictions()

    weights = regime_conditional_weights(predictions, realized, regimes)
    ensemble = regime_conditional_ensemble(predictions, regimes, weights)

    assert set(weights.index) == {"range", "trend"}
    assert ensemble.notna().all()
    assert ensemble.corr(realized) > 0.8


def test_evaluate_prediction_signals_includes_individuals_and_ensemble() -> None:
    predictions, realized, _ = sample_predictions()
    ensemble, _ = ic_weighted_ensemble(predictions, realized, window=20)

    evaluation = evaluate_prediction_signals(predictions, realized, ensemble, ensemble_name="ic_ensemble")

    assert set(evaluation.metrics["model"]) == {"xgboost", "lstm", "linear_baseline", "ic_ensemble"}
    assert "sharpe" in evaluation.metrics.columns
    assert evaluation.ensemble_returns.notna().all()


def test_linear_regression_baseline_predicts_timestamped_series() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC", name="timestamp")
    x = pd.DataFrame({"feature_a": np.arange(40), "feature_b": np.arange(40) * 2}, index=index)
    y = pd.Series(x["feature_a"] * 0.01, index=index)

    model = LinearRegressionBaseline().fit(x.iloc[:30], y.iloc[:30])
    predictions = model.predict(x.iloc[30:])

    assert predictions.name == "linear_baseline"
    assert predictions.index.equals(x.iloc[30:].index)
    assert predictions.corr(y.iloc[30:]) > 0.99
