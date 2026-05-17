"""Feature matrix construction helpers for factor models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features import TargetBuilder


@dataclass(frozen=True)
class FactorDataset:
    """Timestamp-aligned supervised dataset for factor model training."""

    x: pd.DataFrame
    y: pd.Series
    realized_returns: pd.Series


def build_factor_dataset(
    feature_frame: pd.DataFrame,
    target_column: str = "target_return_1",
    target_builder: TargetBuilder | None = None,
    price_column: str = "close",
    feature_columns: list[str] | None = None,
) -> FactorDataset:
    """Build numeric, timestamp-aligned ``X``, ``y``, and realized return series.

    Args:
        feature_frame: DataFrame containing raw OHLCV plus engineered features.
        target_column: Target column to train on. If absent, targets are built.
        target_builder: Optional target builder.
        price_column: Price column used when target construction is needed.
        feature_columns: Optional explicit feature list. If omitted, numeric
            non-price, non-target columns are selected.

    Returns:
        A ``FactorDataset`` with rows dropped only after features and targets
        are joined, preserving timestamp alignment.
    """
    builder = target_builder or TargetBuilder()
    labeled = (
        feature_frame.copy()
        if target_column in feature_frame.columns
        else builder.add_targets(feature_frame, price_column=price_column)
    )

    if feature_columns is None:
        excluded = {"open", "high", "low", "close", "adj_close", "volume"}
        feature_columns = [
            column
            for column in labeled.select_dtypes(include="number").columns
            if column not in excluded and not column.startswith("target_")
        ]

    if target_column not in labeled:
        raise KeyError(f"Target column not found after target construction: {target_column}")

    realized_column = target_column if target_column.startswith("target_return_") else "target_return_1"
    if realized_column not in labeled:
        labeled = builder.add_targets(labeled, price_column=price_column)

    dataset = labeled[feature_columns + [target_column, realized_column]].dropna()
    return FactorDataset(
        x=dataset[feature_columns],
        y=dataset[target_column],
        realized_returns=dataset[realized_column],
    )
