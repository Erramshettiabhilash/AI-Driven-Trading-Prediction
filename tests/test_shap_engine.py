import numpy as np
import pandas as pd

from explainability import (
    analyze_ema_volume_interactions,
    dependence_frame,
    global_feature_importance,
    local_prediction_contributions,
    rank_interactions,
)


def test_global_feature_importance_ranks_mean_absolute_shap_values() -> None:
    shap_values = np.array(
        [
            [0.1, -0.5, 0.2],
            [0.2, 0.3, -0.1],
        ],
    )

    importance = global_feature_importance(shap_values, ["rsi_14", "ema_cross", "volume_zscore"])

    assert importance.iloc[0]["feature"] == "ema_cross"
    assert importance.iloc[0]["importance_rank"] == 1


def test_global_feature_importance_supports_lstm_sequence_values() -> None:
    shap_values = np.zeros((2, 3, 2))
    shap_values[:, :, 0] = 0.1
    shap_values[:, :, 1] = 0.4

    importance = global_feature_importance(shap_values, ["rsi_14", "macd"])

    assert importance.iloc[0]["feature"] == "macd"


def test_local_prediction_contributions_preserve_base_value_metadata() -> None:
    feature_values = pd.Series({"rsi_14": 55.0, "ema_cross": 0.02})
    contributions = local_prediction_contributions(
        np.array([0.01, -0.03]),
        feature_values,
        base_value=0.10,
    )

    assert contributions.iloc[0]["feature"] == "ema_cross"
    assert np.isclose(contributions.attrs["prediction_sum"], 0.08)


def test_dependence_frame_returns_feature_and_interaction_values() -> None:
    features = pd.DataFrame(
        {
            "ema_cross": [0.1, 0.2],
            "volume_zscore": [1.0, 2.0],
        },
    )
    shap_values = np.array([[0.01, 0.02], [0.03, 0.04]])

    output = dependence_frame(
        shap_values,
        features,
        feature="ema_cross",
        interaction_feature="volume_zscore",
    )

    assert output["feature_value"].tolist() == [0.1, 0.2]
    assert output["shap_value"].tolist() == [0.01, 0.03]
    assert output["interaction_value"].tolist() == [1.0, 2.0]


def test_rank_interactions_and_ema_volume_filter() -> None:
    values = np.zeros((3, 3, 3))
    values[:, 0, 1] = 0.2
    values[:, 1, 0] = 0.2
    values[:, 0, 2] = 0.5
    values[:, 2, 0] = 0.5
    names = ["ema_cross", "rsi_14", "volume_zscore"]

    interactions = rank_interactions(values, names)
    ema_volume = analyze_ema_volume_interactions(values, names)

    assert interactions.iloc[0]["feature_b"] == "volume_zscore"
    assert ema_volume.iloc[0]["feature_a"] == "ema_cross"
    assert ema_volume.iloc[0]["feature_b"] == "volume_zscore"
