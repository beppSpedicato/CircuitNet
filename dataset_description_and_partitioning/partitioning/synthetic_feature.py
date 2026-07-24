from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class SyntheticFeaturePartitioner(DatasetPartitioner):
    """Synthetic feature imbalance via design-parameter region split (NIID-Bench strategy 4).

    Inspired by the NIID-Bench "cube division" idea: the 2-D space spanned by
    (``feature_col_x``, ``feature_col_y``) is divided into a grid of cells.
    Each party receives samples whose feature coordinates fall in one or more
    cells.  Within each cell the label distribution remains roughly balanced,
    so only P(X) varies across parties while P(y|x) stays approximately fixed.

    Default axes: utilization × clock_ns — both are continuous design knobs
    that directly affect routing difficulty (the EDA analog of the feature
    space), yet they do not change the underlying label (design correctness).

    Grid size is chosen automatically as the closest integer grid to
    ``n_partitions`` (e.g. 4 → 2×2, 6 → 2×3, 9 → 3×3).  Remaining cells
    are merged into the last party.

    Args:
        n_partitions: Number of federated parties.
        feature_col_x: First feature axis.  Defaults to ``"utilization"``.
        feature_col_y: Second feature axis.  Defaults to ``"clock_ns"``.
        seed: Random seed for shuffling within cells.
    """

    def __init__(
        self,
        n_partitions: int,
        feature_col_x: str = "utilization",
        feature_col_y: str = "clock_ns",
        seed: int = 42,
    ) -> None:
        super().__init__(n_partitions)
        self.feature_col_x = feature_col_x
        self.feature_col_y = feature_col_y
        self.seed = seed

    @staticmethod
    def _grid_shape(n: int) -> Tuple[int, int]:
        """Return (rows, cols) whose product >= n and is as square as possible."""
        rows = int(np.floor(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        return rows, cols

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)
        for col in (self.feature_col_x, self.feature_col_y):
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found in DataFrame")

        df = df.copy().reset_index(drop=True)
        rng = np.random.default_rng(self.seed)

        rows, cols = self._grid_shape(self.n_partitions)

        # Quantile-based bin edges so cells are equally populated.
        x_vals = df[self.feature_col_x].values.astype(float)
        y_vals = df[self.feature_col_y].values.astype(float)

        x_edges = np.quantile(x_vals, np.linspace(0, 1, cols + 1))
        y_edges = np.quantile(y_vals, np.linspace(0, 1, rows + 1))

        # Make boundary edges inclusive.
        x_edges[0] -= 1e-9
        x_edges[-1] += 1e-9
        y_edges[0] -= 1e-9
        y_edges[-1] += 1e-9

        x_bin = np.searchsorted(x_edges[1:], x_vals, side="left")
        y_bin = np.searchsorted(y_edges[1:], y_vals, side="left")
        cell_id = y_bin * cols + x_bin  # linear cell index

        # Map cell_id → party (cells beyond n_partitions merge into last party).
        cell_to_party = np.minimum(cell_id, self.n_partitions - 1)

        partitions = []
        for i in range(self.n_partitions):
            mask = cell_to_party == i
            part = df[mask].copy()
            idx = part.index.tolist()
            rng.shuffle(idx)
            partitions.append(part.loc[idx].reset_index(drop=True))
        return partitions
