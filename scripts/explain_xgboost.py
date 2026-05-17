"""Generate SHAP explainability artifacts for a trained XGBoost factor model.

Example:
    python scripts/explain_xgboost.py ^
        --model results/xgboost/xgboost_regression.json ^
        --features-json results/xgboost/xgboost_regression_features.json ^
        --input data/processed/SPY_features.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import pandas as pd

from explainability import ShapExplainer, dependence_frame, global_feature_importance
from models import XGBoostFactorModel, build_factor_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for XGBoost SHAP reporting."""
    parser = argparse.ArgumentParser(description="Explain a trained XGBoost model with SHAP.")
    parser.add_argument("--model", required=True, help="Path to trained XGBoost JSON model.")
    parser.add_argument("--features-json", required=True, help="JSON list of model feature names.")
    parser.add_argument("--input", required=True, help="Feature CSV with timestamp column.")
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Original model task.",
    )
    parser.add_argument("--target", default="target_return_1", help="Target column for dataset rebuild.")
    parser.add_argument("--max-rows", type=int, default=1000, help="Maximum rows to explain.")
    parser.add_argument("--output-dir", default="results/explainability", help="Artifact directory.")
    return parser.parse_args()


def main() -> None:
    """Load model/data, compute SHAP values, and save explainability tables."""
    args = parse_args()
    feature_names = json.loads(Path(args.features_json).read_text(encoding="utf-8"))

    frame = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    dataset = build_factor_dataset(frame, target_column=args.target, feature_columns=feature_names)
    x = dataset.x.tail(args.max_rows)

    model = XGBoostFactorModel(task=args.task).load_model(args.model, feature_names=feature_names)
    explanation = ShapExplainer().explain_xgboost(model, x)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    importance_path = output_dir / "xgboost_global_importance.csv"
    interactions_path = output_dir / "xgboost_interactions.csv"
    dependence_path = output_dir / "xgboost_top_dependence.csv"

    importance = global_feature_importance(explanation.values, explanation.feature_names)
    importance.to_csv(importance_path, index=False)

    top_feature = str(importance.iloc[0]["feature"])
    dependence_frame(explanation.values, x, feature=top_feature).to_csv(dependence_path)

    try:
        interactions = ShapExplainer().xgboost_interactions(model, x)
        interactions.to_csv(interactions_path, index=False)
    except Exception as exc:  # pragma: no cover - optional artifact path
        interactions_path.write_text(f"Interaction calculation failed: {exc}", encoding="utf-8")

    print(
        json.dumps(
            {
                "importance": str(importance_path),
                "dependence": str(dependence_path),
                "interactions": str(interactions_path),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
