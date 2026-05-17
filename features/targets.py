"""Target construction for financial machine learning research.

Targets are labels, not tradable signals. They intentionally look forward
because they describe what the model is trying to predict. The surrounding
feature and validation code must ensure that features at time ``t`` contain
only information available at or before ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetBuilder:
    """Create regression and classification targets from close prices."""

    horizons: tuple[int, ...] = (1, 5)
    min_history_for_tertiles: int = 60

    def add_targets(self, frame: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """Add all Step 4 targets for configured horizons.

        For each horizon ``N`` the method creates:

        - ``target_return_N``: next ``N``-period log return.
        - ``target_direction_N``: ``1`` for non-negative return, ``-1`` otherwise.
        - ``target_tertile_N``: ``-1`` bottom tertile, ``0`` middle, ``1`` top.

        Args:
            frame: Input DataFrame containing a close price column.
            price_column: Price column used to calculate forward returns.

        Returns:
            A copy of ``frame`` with target columns appended.
        """
        if price_column not in frame:
            raise KeyError(f"Missing price column for target construction: {price_column}")

        output = frame.copy()
        for horizon in self.horizons:
            target_column = f"target_return_{horizon}"
            output[target_column] = self.forward_log_return(output[price_column], horizon)
            output[f"target_direction_{horizon}"] = self.direction_label(output[target_column])
            output[f"target_tertile_{horizon}"] = self.expanding_tertile_label(
                output[target_column],
                min_history=self.min_history_for_tertiles,
            )

        return output

    @staticmethod
    def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
        """Return the next ``horizon``-period log return.

        The label at timestamp ``t`` is ``log(close[t + horizon] / close[t])``.
        The final ``horizon`` rows are unknown and therefore ``NaN``.
        """
        if horizon <= 0:
            raise ValueError("Target horizon must be positive.")
        return np.log(close.shift(-horizon) / close)

    @staticmethod
    def direction_label(forward_return: pd.Series) -> pd.Series:
        """Return ``1`` for non-negative forward returns and ``-1`` for negative ones."""
        label = pd.Series(np.where(forward_return >= 0, 1, -1), index=forward_return.index)
        return label.where(forward_return.notna())

    @staticmethod
    def expanding_tertile_label(
        forward_return: pd.Series,
        min_history: int = 60,
    ) -> pd.Series:
        """Bucket forward returns into expanding, shifted tertiles.

        Full-sample tertiles use the future return distribution and can make
        research cleaner than reality. This implementation uses thresholds
        from prior labeled observations only.
        """
        lower = forward_return.expanding(min_periods=min_history).quantile(1 / 3).shift(1)
        upper = forward_return.expanding(min_periods=min_history).quantile(2 / 3).shift(1)
        labels = pd.Series(np.nan, index=forward_return.index, dtype="float64")
        labels = labels.mask(forward_return <= lower, -1)
        labels = labels.mask((forward_return > lower) & (forward_return < upper), 0)
        labels = labels.mask(forward_return >= upper, 1)
        return labels


def build_supervised_frame(
    feature_frame: pd.DataFrame,
    target_builder: TargetBuilder | None = None,
    price_column: str = "close",
    feature_columns: Iterable[str] | None = None,
    target_column: str = "target_return_1",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return aligned ``X`` and ``y`` for supervised financial ML.

    Rows with missing features or missing target labels are dropped together so
    timestamps stay aligned.
    """
    builder = target_builder or TargetBuilder()
    labeled = builder.add_targets(feature_frame, price_column=price_column)

    if feature_columns is None:
        excluded_prefixes = ("target_",)
        excluded_columns = {"open", "high", "low", "close", "adj_close", "volume"}
        feature_columns = [
            column
            for column in labeled.columns
            if column not in excluded_columns and not column.startswith(excluded_prefixes)
        ]

    dataset = labeled[list(feature_columns) + [target_column]].dropna()
    return dataset[list(feature_columns)], dataset[target_column]
