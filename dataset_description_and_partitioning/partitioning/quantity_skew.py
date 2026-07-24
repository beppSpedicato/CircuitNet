from typing import List, Optional

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class QuantitySkewPartitioner(DatasetPartitioner):
    """Quantity skew via Dirichlet-sampled partition sizes (NIID-Bench strategy 5).

    The total dataset size is fixed, but each party's share is drawn from a
    Dirichlet(alpha) distribution, producing highly unequal partition sizes.
    The label composition within each party mirrors the global distribution
    (no label skew), so only the *amount* of data varies.

    Lower alpha → more extreme size imbalance (one party may hold most of
    the data while others get very little).  alpha → ∞ → equal sizes (IID
    quantity regime).

    Args:
        n_partitions: Number of federated parties.
        alpha: Dirichlet concentration.  Recommended range: 0.1 – 10.
        min_samples_per_party: Hard floor to avoid empty partitions.
        seed: Random seed.
    """

    def __init__(
        self,
        n_partitions: int,
        alpha: float = 0.5,
        min_samples_per_party: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__(n_partitions)
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.min_samples_per_party = min_samples_per_party
        self.seed = seed

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)

        df = df.copy().reset_index(drop=True)
        rng = np.random.default_rng(self.seed)
        n = len(df)

        # Sample proportions until the floor constraint is met.
        for _ in range(1000):
            props = rng.dirichlet([self.alpha] * self.n_partitions)
            counts = (props * n).astype(int)
            counts[np.argmax(props)] += n - counts.sum()
            if counts.min() >= self.min_samples_per_party:
                break

        # Shuffle once, then assign contiguous slices.
        shuffled_idx = rng.permutation(n)
        partitions = []
        cursor = 0
        for cnt in counts:
            indices = shuffled_idx[cursor : cursor + cnt]
            partitions.append(df.iloc[indices].reset_index(drop=True))
            cursor += cnt
        return partitions
