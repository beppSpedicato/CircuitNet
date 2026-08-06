from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class FeatureDirichletPartitioner(DatasetPartitioner):
    """P2 — Feature Dirichlet: metadata-value Dirichlet allocation.

    For each unique value of the composite key formed by ``feature_cols``,
    draws a Dirichlet(alpha, ..., alpha) vector of length ``n_partitions``
    and splits that value's samples across clients according to the
    resulting proportions. This gives a continuous non-IID knob on the
    metadata marginal P(X) without touching P(y|x).

    - alpha -> 0: each metadata value goes almost entirely to a single
      client (highly specialized clients, extreme non-IID).
    - alpha -> +inf: each value split evenly across clients (recovers IID).

    Contrast with :class:`~partitioning.dirichlet_label.DirichletLabelPartitioner`,
    which drives skew from the DRC severity *label* (tier). Here the skew
    comes from a METADATA attribute chosen to match the primary axis
    identified in Phase 2/3 (typically ``design_name`` on N28, optionally
    combined with a persona bin).

    Args:
        n_partitions: Number of federated parties.
        feature_cols: Column name or list of column names forming the
            composite key. All must be present in the input DataFrame.
        alpha: Dirichlet concentration. Recommended sweep: 0.1, 0.5, 5.0.
        min_samples_per_party: Redraw the Dirichlet for a given key value
            until every party's per-value share meets this floor. Set to
            0 to allow parties with zero samples for a value.
        max_resample_tries: Cap on the resampling loop per key value.
        seed: Random seed.
    """

    def __init__(
        self,
        n_partitions: int,
        feature_cols: Union[str, Sequence[str]] = "design_name",
        alpha: float = 0.5,
        min_samples_per_party: int = 0,
        max_resample_tries: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__(n_partitions)
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if min_samples_per_party < 0:
            raise ValueError("min_samples_per_party must be >= 0")
        self.feature_cols = (
            [feature_cols] if isinstance(feature_cols, str) else list(feature_cols)
        )
        self.alpha = float(alpha)
        self.min_samples_per_party = int(min_samples_per_party)
        self.max_resample_tries = int(max_resample_tries)
        self.seed = int(seed)

    def _composite_key(self, df: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"feature_cols missing from DataFrame: {missing}"
            )
        return df[self.feature_cols].astype(str).agg("|".join, axis=1)

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)
        df = df.copy().reset_index(drop=True)
        rng = np.random.default_rng(self.seed)

        keys = self._composite_key(df)
        party_indices: List[List[int]] = [[] for _ in range(self.n_partitions)]

        # Iterate over sorted unique keys for determinism given the same seed.
        for key_value in sorted(keys.unique()):
            key_idx = df.index[keys == key_value].to_numpy()
            rng.shuffle(key_idx)
            n = len(key_idx)

            # Cap the per-value floor by what a value can actually supply.
            effective_floor = min(self.min_samples_per_party, n // self.n_partitions)

            for _ in range(self.max_resample_tries):
                props = rng.dirichlet([self.alpha] * self.n_partitions)
                counts = (props * n).astype(int)
                counts[np.argmax(props)] += n - counts.sum()
                if effective_floor == 0 or counts.min() >= effective_floor:
                    break

            cursor = 0
            for i, cnt in enumerate(counts):
                party_indices[i].extend(key_idx[cursor : cursor + cnt].tolist())
                cursor += cnt

        return [
            df.loc[party_indices[i]].reset_index(drop=True)
            for i in range(self.n_partitions)
        ]
