"""SHAP explainability utilities for model governance and research review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShapExplanation:
    """Container for SHAP values aligned to feature data."""

    values: np.ndarray
    base_values: np.ndarray | float | None
    data: pd.DataFrame | np.ndarray
    feature_names: list[str]


def _as_primary_shap_array(values: Any) -> np.ndarray:
    """Normalize SHAP outputs into a numeric numpy array.

    SHAP can return lists for multi-class models and explanation objects for
    newer APIs. For binary classifiers, the positive class is usually the most
    useful trading explanation, so the final class array is selected.
    """
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        values = values[-1]
    values_array = np.asarray(values)
    if values_array.ndim >= 3 and values_array.shape[-1] == 1:
        values_array = values_array[..., 0]
    return values_array


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    For sequence models with shape ``samples x timesteps x features``, the
    importance is averaged across samples and timesteps.
    """
    values = _as_primary_shap_array(shap_values)
    if values.ndim == 2:
        importance = np.abs(values).mean(axis=0)
    elif values.ndim == 3:
        importance = np.abs(values).mean(axis=(0, 1))
    else:
        raise ValueError("SHAP values must be 2D or 3D.")

    if len(importance) != len(feature_names):
        raise ValueError("Number of feature names does not match SHAP value width.")

    output = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        },
    )
    output["importance_rank"] = output["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    return output.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_prediction_contributions(
    shap_values: np.ndarray,
    feature_values: pd.Series,
    base_value: float | None = None,
) -> pd.DataFrame:
    """Return per-feature contributions for one prediction."""
    values = _as_primary_shap_array(shap_values)
    if values.ndim != 1:
        raise ValueError("local_prediction_contributions expects one row of SHAP values.")
    if len(values) != len(feature_values):
        raise ValueError("Feature values and SHAP values must have the same length.")

    output = pd.DataFrame(
        {
            "feature": feature_values.index,
            "feature_value": feature_values.to_numpy(),
            "shap_value": values,
            "abs_shap": np.abs(values),
        },
    ).sort_values("abs_shap", ascending=False)

    if base_value is not None:
        output.attrs["base_value"] = float(base_value)
        output.attrs["prediction_sum"] = float(base_value + values.sum())

    return output.reset_index(drop=True)


def dependence_frame(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature: str,
    interaction_feature: str | None = None,
) -> pd.DataFrame:
    """Return data needed for a SHAP dependence plot."""
    values = _as_primary_shap_array(shap_values)
    if values.ndim != 2:
        raise ValueError("dependence_frame expects 2D tabular SHAP values.")
    if feature not in features:
        raise KeyError(f"Unknown feature: {feature}")

    feature_index = list(features.columns).index(feature)
    output = pd.DataFrame(
        {
            "feature": feature,
            "feature_value": features[feature].to_numpy(),
            "shap_value": values[:, feature_index],
        },
        index=features.index,
    )

    if interaction_feature is not None:
        if interaction_feature not in features:
            raise KeyError(f"Unknown interaction feature: {interaction_feature}")
        output["interaction_feature"] = interaction_feature
        output["interaction_value"] = features[interaction_feature].to_numpy()

    return output


def rank_interactions(
    shap_interaction_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank pairwise SHAP interactions by mean absolute interaction value."""
    values = _as_primary_shap_array(shap_interaction_values)
    if values.ndim != 3:
        raise ValueError("Interaction values must have shape samples x features x features.")
    if values.shape[1] != len(feature_names) or values.shape[2] != len(feature_names):
        raise ValueError("Feature names do not match interaction value dimensions.")

    rows: list[dict[str, object]] = []
    for left in range(len(feature_names)):
        for right in range(left + 1, len(feature_names)):
            rows.append(
                {
                    "feature_a": feature_names[left],
                    "feature_b": feature_names[right],
                    "mean_abs_interaction": float(np.abs(values[:, left, right]).mean()),
                },
            )

    return pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)


def analyze_ema_volume_interactions(
    shap_interaction_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return EMA and volume interaction strengths from SHAP interaction values."""
    interactions = rank_interactions(shap_interaction_values, feature_names)
    ema_volume_mask = interactions.apply(
        lambda row: (
            ("ema" in str(row["feature_a"]).lower() and "volume" in str(row["feature_b"]).lower())
            or ("volume" in str(row["feature_a"]).lower() and "ema" in str(row["feature_b"]).lower())
        ),
        axis=1,
    )
    return interactions.loc[ema_volume_mask].reset_index(drop=True)


class ShapExplainer:
    """SHAP explainability engine for XGBoost and LSTM models."""

    def explain_xgboost(
        self,
        model: Any,
        features: pd.DataFrame,
        model_output: str = "raw",
    ) -> ShapExplanation:
        """Compute TreeExplainer SHAP values for an XGBoost model or wrapper."""
        shap = self._shap()
        booster = getattr(model, "booster_", model)
        explainer = shap.TreeExplainer(booster, model_output=model_output)
        values = _as_primary_shap_array(explainer.shap_values(features))
        return ShapExplanation(
            values=values,
            base_values=getattr(explainer, "expected_value", None),
            data=features,
            feature_names=list(features.columns),
        )

    def explain_lstm(
        self,
        model: Any,
        background_sequences: np.ndarray,
        sample_sequences: np.ndarray,
        feature_names: list[str],
    ) -> ShapExplanation:
        """Compute DeepExplainer SHAP values for an LSTM model or wrapper."""
        shap = self._shap()
        keras_model = getattr(model, "model_", model)
        explainer = shap.DeepExplainer(keras_model, background_sequences)
        values = _as_primary_shap_array(explainer.shap_values(sample_sequences))
        return ShapExplanation(
            values=values,
            base_values=getattr(explainer, "expected_value", None),
            data=sample_sequences,
            feature_names=feature_names,
        )

    def xgboost_interactions(self, model: Any, features: pd.DataFrame) -> pd.DataFrame:
        """Compute and rank XGBoost SHAP interaction values."""
        shap = self._shap()
        booster = getattr(model, "booster_", model)
        explainer = shap.TreeExplainer(booster)
        interaction_values = explainer.shap_interaction_values(features)
        return rank_interactions(interaction_values, list(features.columns))

    def save_force_plot(
        self,
        explanation: ShapExplanation,
        row: int,
        output_path: str | Path,
    ) -> Path:
        """Save a SHAP force plot HTML file for one prediction."""
        shap = self._shap()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        values = _as_primary_shap_array(explanation.values)
        if values.ndim != 2:
            raise ValueError("Force plots currently support 2D tabular SHAP values.")
        data = explanation.data.iloc[row] if isinstance(explanation.data, pd.DataFrame) else explanation.data[row]
        base_value = self._base_value_for_row(explanation.base_values, row)
        force_plot = shap.force_plot(base_value, values[row], data, feature_names=explanation.feature_names)
        shap.save_html(str(output), force_plot)
        return output

    def save_dependence_plot(
        self,
        explanation: ShapExplanation,
        feature: str,
        output_path: str | Path,
        interaction_feature: str | None = None,
    ) -> Path:
        """Save a SHAP dependence plot image for a tabular explanation."""
        shap = self._shap()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(explanation.data, pd.DataFrame):
            raise ValueError("Dependence plots require tabular feature data.")

        import matplotlib.pyplot as plt

        shap.dependence_plot(
            feature,
            _as_primary_shap_array(explanation.values),
            explanation.data,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()
        return output

    @staticmethod
    def _base_value_for_row(base_values: np.ndarray | float | None, row: int) -> float:
        """Return a scalar base value for one prediction row."""
        if base_values is None:
            return 0.0
        base_array = np.asarray(base_values)
        if base_array.ndim == 0:
            return float(base_array)
        if len(base_array) == 1:
            return float(base_array[0])
        return float(base_array[row])

    @staticmethod
    def _shap() -> Any:
        """Import SHAP with a clear setup error."""
        try:
            import shap
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install shap to use ShapExplainer.") from exc
        return shap
