from typing import List, Optional

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class DirichletLabelPartitioner(DatasetPartitioner):
    """Distribution-based label imbalance via Dirichlet (NIID-Bench strategy 2).

    For each label class, a Dirichlet(alpha) draw determines the proportion of
    that class's samples allocated to each party.  Lower alpha → more skewed
    label distributions across parties.  alpha → ∞ recovers IID behavior.

    Args:
        n_partitions: Number of federated parties.
        alpha: Dirichlet concentration parameter.  Common choices: 0.1 (very
            skewed), 0.5 (moderately skewed), 100 (near-IID).
        label_col: Column used as the class label.
        min_samples_per_party: Minimum samples any party must receive.  Tiny
            Dirichlet draws are re-sampled until this floor is met.
        seed: Random seed.
    """

    def __init__(
        self,
        n_partitions: int,
        alpha: float = 0.5,
        label_col: str = "design_name",
        min_samples_per_party: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__(n_partitions)
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.label_col = label_col
        self.min_samples_per_party = min_samples_per_party
        self.seed = seed

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)
        if self.label_col not in df.columns:
            raise ValueError(f"label_col '{self.label_col}' not found in DataFrame")

        rng = np.random.default_rng(self.seed)
        df = df.copy().reset_index(drop=True)
        party_indices: List[List[int]] = [[] for _ in range(self.n_partitions)]

        for lbl in df[self.label_col].unique():
            lbl_idx = df.index[df[self.label_col] == lbl].tolist()
            rng.shuffle(lbl_idx)
            n = len(lbl_idx)

            # Draw proportions until every party gets >= min_samples floor.
            for _ in range(1000):
                props = rng.dirichlet([self.alpha] * self.n_partitions)
                counts = (props * n).astype(int)
                # Distribute rounding remainder to the largest-proportion party.
                remainder = n - counts.sum()
                counts[np.argmax(props)] += remainder
                if counts.min() >= self.min_samples_per_party:
                    break

            cursor = 0
            for i, cnt in enumerate(counts):
                party_indices[i].extend(lbl_idx[cursor : cursor + cnt])
                cursor += cnt

        return [
            df.loc[party_indices[i]].reset_index(drop=True)
            for i in range(self.n_partitions)
        ]
