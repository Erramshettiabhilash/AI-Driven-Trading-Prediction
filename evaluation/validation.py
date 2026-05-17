"""Time-series validation utilities with purging and embargo support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


IndexArray = np.ndarray


def _resolve_test_size(n_samples: int, test_size: int | float) -> int:
    """Convert a fractional or absolute test size into a positive sample count."""
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Float test_size must be between 0 and 1.")
        resolved = int(np.ceil(n_samples * test_size))
    else:
        resolved = int(test_size)

    if resolved <= 0 or resolved >= n_samples:
        raise ValueError("test_size must leave at least one train and one test row.")
    return resolved


def temporal_train_test_split(
    data: pd.DataFrame | pd.Series,
    test_size: int | float = 0.2,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> tuple[IndexArray, IndexArray]:
    """Return chronological train/test integer indices.

    Args:
        data: Time-ordered data.
        test_size: Number of test rows, or fraction of all rows.
        purge_bars: Rows removed from the end of training before test begins.
        embargo_bars: Rows removed from the beginning of testing after train ends.

    Returns:
        ``(train_indices, test_indices)`` as numpy integer arrays.
    """
    n_samples = len(data)
    resolved_test_size = _resolve_test_size(n_samples, test_size)
    split_at = n_samples - resolved_test_size

    train_end = max(0, split_at - purge_bars)
    test_start = min(n_samples, split_at + embargo_bars)
    if train_end == 0 or test_start >= n_samples:
        raise ValueError("purge_bars/embargo_bars removed all train or test rows.")

    return np.arange(0, train_end), np.arange(test_start, n_samples)


@dataclass(frozen=True)
class ExpandingWindowSplit:
    """Expanding-window cross-validation for time-series data."""

    initial_train_size: int
    test_size: int
    step_size: int | None = None
    purge_bars: int = 0
    embargo_bars: int = 0

    def split(self, data: pd.DataFrame | pd.Series) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Yield chronological expanding train/test splits."""
        n_samples = len(data)
        step = self.step_size or self.test_size
        train_end = self.initial_train_size

        while train_end + self.embargo_bars < n_samples:
            effective_train_end = train_end - self.purge_bars
            test_start = train_end + self.embargo_bars
            test_end = min(test_start + self.test_size, n_samples)

            if effective_train_end <= 0 or test_start >= test_end:
                break

            yield np.arange(0, effective_train_end), np.arange(test_start, test_end)
            if test_end == n_samples:
                break
            train_end += step


@dataclass(frozen=True)
class RollingWindowSplit:
    """Rolling-window cross-validation with fixed-size training windows."""

    train_size: int
    test_size: int
    step_size: int | None = None
    purge_bars: int = 0
    embargo_bars: int = 0

    def split(self, data: pd.DataFrame | pd.Series) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Yield chronological rolling train/test splits."""
        n_samples = len(data)
        step = self.step_size or self.test_size
        train_start = 0

        while train_start + self.train_size + self.embargo_bars < n_samples:
            train_end = train_start + self.train_size
            effective_train_end = train_end - self.purge_bars
            test_start = train_end + self.embargo_bars
            test_end = min(test_start + self.test_size, n_samples)

            if effective_train_end <= train_start or test_start >= test_end:
                break

            yield np.arange(train_start, effective_train_end), np.arange(test_start, test_end)
            if test_end == n_samples:
                break
            train_start += step
