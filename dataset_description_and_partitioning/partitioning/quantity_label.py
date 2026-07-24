from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class QuantityLabelPartitioner(DatasetPartitioner):
    """Quantity-based label imbalance (NIID-Bench strategy 1).

    Each party is assigned exactly *k* label classes.  For every class, its
    samples are split equally among the parties that own that class.  The
    assignment guarantees no sample overlap across parties.

    Args:
        n_partitions: Number of federated parties.
        n_labels_per_party: How many distinct label values each party owns.
            Must satisfy 1 <= n_labels_per_party <= n_unique_labels.
        label_col: Column used as the class label.  Defaults to
            ``"design_name"``, the six base RTL designs in CircuitNet-N28.
        seed: Random seed for reproducible assignment.
    """

    def __init__(
        self,
        n_partitions: int,
        n_labels_per_party: int = 2,
        label_col: str = "design_name",
        seed: int = 42,
    ) -> None:
        super().__init__(n_partitions)
        if n_labels_per_party < 1:
            raise ValueError("n_labels_per_party must be >= 1")
        self.n_labels_per_party = n_labels_per_party
        self.label_col = label_col
        self.seed = seed

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)
        if self.label_col not in df.columns:
            raise ValueError(f"label_col '{self.label_col}' not found in DataFrame")

        rng = np.random.default_rng(self.seed)
        labels = df[self.label_col].unique().tolist()
        n_labels = len(labels)

        if self.n_labels_per_party > n_labels:
            raise ValueError(
                f"n_labels_per_party ({self.n_labels_per_party}) exceeds the "
                f"number of unique labels ({n_labels})"
            )

        # Round-robin assignment of labels to parties so that every label is
        # owned by at least one party.
        shuffled_labels = rng.permutation(labels).tolist()
        party_labels: Dict[int, List] = {i: [] for i in range(self.n_partitions)}
        label_pool = (shuffled_labels * self.n_partitions)[
            : self.n_partitions * self.n_labels_per_party
        ]
        rng.shuffle(label_pool)
        for party, lbl in zip(
            [i for i in range(self.n_partitions) for _ in range(self.n_labels_per_party)],
            label_pool,
        ):
            if lbl not in party_labels[party]:
                party_labels[party].append(lbl)

        # Ensure every party has exactly n_labels_per_party labels.
        for i in range(self.n_partitions):
            while len(party_labels[i]) < self.n_labels_per_party:
                candidate = rng.choice(labels)
                if candidate not in party_labels[i]:
                    party_labels[i].append(candidate)

        # For each label, split its samples equally among owning parties.
        party_indices: Dict[int, List[int]] = {i: [] for i in range(self.n_partitions)}
        for lbl in labels:
            owners = [i for i in range(self.n_partitions) if lbl in party_labels[i]]
            if not owners:
                owners = list(range(self.n_partitions))
            lbl_idx = df.index[df[self.label_col] == lbl].tolist()
            rng.shuffle(lbl_idx)
            chunks = np.array_split(lbl_idx, len(owners))
            for owner, chunk in zip(owners, chunks):
                party_indices[owner].extend(chunk.tolist())

        return [
            df.loc[party_indices[i]].reset_index(drop=True)
            for i in range(self.n_partitions)
        ]
