"""Explainability package for financial model governance."""

from explainability.shap_engine import (
    ShapExplainer,
    ShapExplanation,
    analyze_ema_volume_interactions,
    dependence_frame,
    global_feature_importance,
    local_prediction_contributions,
    rank_interactions,
)

__all__ = [
    "ShapExplainer",
    "ShapExplanation",
    "analyze_ema_volume_interactions",
    "dependence_frame",
    "global_feature_importance",
    "local_prediction_contributions",
    "rank_interactions",
]
