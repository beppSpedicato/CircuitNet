from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class NoiseFeaturePartitioner(DatasetPartitioner):
    """Noise-based feature imbalance (NIID-Bench strategy 3).

    The dataset is first divided into equal-sized IID splits (stratified by
    design configuration), then each party is tagged with a per-party Gaussian
    noise standard deviation stored in the ``noise_std`` column.  Actual noise
    injection happens at training time when the features are loaded.

    The noise levels either follow a linear ramp from *noise_std_min* to
    *noise_std_max*, or are provided explicitly as a list.

    Args:
        n_partitions: Number of federated parties.
        noise_std_min: Minimum Gaussian noise std (party 0).
        noise_std_max: Maximum Gaussian noise std (party n-1).
        noise_stds: Explicit per-party noise stds; overrides min/max if given.
        stratify_cols: Columns used for the IID base split.
        seed: Random seed.
    """

    _DEFAULT_STRATIFY_COLS: List[str] = [
        "design_name",
        "macro_count",
        "macro_placement",
        "power_mesh",
        "filler_insertion",
    ]

    def __init__(
        self,
        n_partitions: int,
        noise_std_min: float = 0.0,
        noise_std_max: float = 0.5,
        noise_stds: Optional[List[float]] = None,
        stratify_cols: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(n_partitions)
        if noise_stds is not None:
            if len(noise_stds) != n_partitions:
                raise ValueError(
                    f"noise_stds length ({len(noise_stds)}) must equal "
                    f"n_partitions ({n_partitions})"
                )
            self.noise_stds = noise_stds
        else:
            self.noise_stds = list(
                np.linspace(noise_std_min, noise_std_max, n_partitions)
            )
        self.stratify_cols = stratify_cols or self._DEFAULT_STRATIFY_COLS
        self.seed = seed

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)

        df = df.copy().reset_index(drop=True)
        rng = np.random.default_rng(self.seed)

        available = [c for c in self.stratify_cols if c in df.columns]
        strat_key = (
            df[available].astype(str).agg("-".join, axis=1)
            if available
            else pd.Series(["_all_"] * len(df), index=df.index)
        )

        assignment = np.empty(len(df), dtype=int)
        for group_positions in df.groupby(strat_key).groups.values():
            positions = np.array(group_positions)
            rng.shuffle(positions)
            assignment[positions] = np.arange(len(positions)) % self.n_partitions

        partitions = []
        for i in range(self.n_partitions):
            part = df[assignment == i].copy().reset_index(drop=True)
            part["noise_std"] = self.noise_stds[i]
            partitions.append(part)
        return partitions
