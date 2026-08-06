from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .base import DatasetPartitioner


# (column, kind, weight) tuple used to describe one input feature.
# kind is one of {"nominal", "ordinal", "interval"}.
FeatureSpec = Tuple[str, str, float]


class KPrototypesPartitioner(DatasetPartitioner):
    """P3 — Data-driven partitioner via weighted-Gower proxy + k-means.

    Encodes mixed-type metadata into a single weighted numeric space that
    approximates weighted Gower dissimilarity under squared-Euclidean
    distance, then runs k-means (n_clusters = n_partitions). Each cluster
    label becomes a client id, so the partitioning is *discovered* rather
    than imposed.

    Encoding per feature type:
        - nominal  : one-hot columns; the whole block for feature k is
                     scaled by sqrt(weight_k / 2), so squared Euclidean
                     between two samples equals weight_k when their level
                     differs and 0 when equal (matching Gower's 0/1
                     nominal contribution, weighted).
        - ordinal  : mapped to normalized rank in [0, 1] using the order
                     supplied via ordinal_orders[col]; scaled by
                     sqrt(weight_k).
        - interval : min-max normalized to [0, 1]; scaled by sqrt(weight_k).

    The encoded space is thus a per-feature-weighted Euclidean surrogate
    for Gower distance (Gower uses L1 in the numeric part; the L2 version
    is a monotonic proxy widely used in practice and lets us use k-means).

    Features listed in ``feature_specs`` that are missing from the input
    DataFrame, or that have a single unique value, are dropped with a
    warning-free note in ``self.dropped_features_`` after ``partition``
    is called -- this keeps the same spec usable across N28 (no
    aspect_ratio, near-constant size_class) and N14 (all present).

    Comparing the cluster labels emitted here against a P1
    HierarchicalPersonaPartitioner assignment (via ARI / NMI) is the
    validation step: high agreement -> hand-drawn personas match the
    natural metadata structure; low agreement -> the data suggests a
    different grouping.

    Args:
        n_partitions: Number of federated parties (== k for k-means).
        feature_specs: Sequence of (column_name, kind, weight) tuples.
            kind is one of {"nominal", "ordinal", "interval"}.
        ordinal_orders: Mapping column_name -> ordered list of category
            levels (low to high). Required for every ordinal feature.
        seed: Random seed for k-means init.
        n_init: k-means restarts. Higher = more stable clusters.
        max_iter: k-means maximum iterations per init.
        drop_missing: If True (default), silently drop features absent
            from the input DataFrame. If False, raise a ValueError.
        drop_constant: If True (default), drop features with a single
            unique value (they contribute nothing to distance).
    """

    _VALID_KINDS = frozenset({"nominal", "ordinal", "interval"})

    def __init__(
        self,
        n_partitions: int,
        feature_specs: Sequence[FeatureSpec],
        ordinal_orders: Optional[Dict[str, Sequence]] = None,
        seed: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        drop_missing: bool = True,
        drop_constant: bool = True,
    ) -> None:
        super().__init__(n_partitions)
        if not feature_specs:
            raise ValueError("feature_specs must be non-empty")
        for col, kind, weight in feature_specs:
            if kind not in self._VALID_KINDS:
                raise ValueError(
                    f"Unknown feature kind '{kind}' for column '{col}'. "
                    f"Must be one of {sorted(self._VALID_KINDS)}."
                )
            if weight <= 0:
                raise ValueError(f"weight for '{col}' must be > 0, got {weight}")
        self.feature_specs: List[FeatureSpec] = [
            (col, kind, float(w)) for col, kind, w in feature_specs
        ]
        self.ordinal_orders: Dict[str, List] = {
            col: list(levels) for col, levels in (ordinal_orders or {}).items()
        }
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.drop_missing = bool(drop_missing)
        self.drop_constant = bool(drop_constant)

        self.dropped_features_: List[Tuple[str, str]] = []
        self.encoded_matrix_: Optional[np.ndarray] = None
        self.cluster_labels_: Optional[np.ndarray] = None
        self.kmeans_: Optional[KMeans] = None

    def _prepare_specs(self, df: pd.DataFrame) -> List[FeatureSpec]:
        prepared: List[FeatureSpec] = []
        self.dropped_features_ = []
        for col, kind, weight in self.feature_specs:
            if col not in df.columns:
                if self.drop_missing:
                    self.dropped_features_.append((col, "missing"))
                    continue
                raise ValueError(f"Feature '{col}' missing from DataFrame")
            if self.drop_constant and df[col].nunique(dropna=True) <= 1:
                self.dropped_features_.append((col, "constant"))
                continue
            if kind == "ordinal" and col not in self.ordinal_orders:
                raise ValueError(
                    f"Ordinal feature '{col}' requires an entry in ordinal_orders"
                )
            prepared.append((col, kind, weight))
        if not prepared:
            raise ValueError(
                "All feature_specs were dropped (missing or constant). "
                f"Dropped: {self.dropped_features_}"
            )
        return prepared

    def _encode_nominal(self, series: pd.Series, weight: float) -> np.ndarray:
        levels = sorted(series.dropna().unique().tolist(), key=str)
        if len(levels) <= 1:
            return np.zeros((len(series), 0), dtype=float)
        level_to_idx = {lvl: i for i, lvl in enumerate(levels)}
        idx = series.map(level_to_idx).to_numpy()
        onehot = np.zeros((len(series), len(levels)), dtype=float)
        valid = ~pd.isna(series.to_numpy())
        onehot[np.arange(len(series))[valid], idx[valid].astype(int)] = 1.0
        # Scale so that mismatch contributes squared distance = weight.
        # Two one-hot vectors at different positions differ by sqrt(2) in
        # L2; squaring gives 2. Multiplying by sqrt(weight/2) rescales
        # this to weight.
        return onehot * np.sqrt(weight / 2.0)

    def _encode_ordinal(self, series: pd.Series, weight: float, order: Sequence) -> np.ndarray:
        rank_map = {lvl: i for i, lvl in enumerate(order)}
        unseen = set(series.dropna().unique()) - set(rank_map)
        if unseen:
            raise ValueError(
                f"Ordinal column has values outside its declared order: {sorted(unseen)}"
            )
        ranks = series.map(rank_map).to_numpy(dtype=float)
        if len(order) <= 1:
            return np.zeros((len(series), 1), dtype=float)
        normed = ranks / (len(order) - 1)
        return (normed * np.sqrt(weight)).reshape(-1, 1)

    def _encode_interval(self, series: pd.Series, weight: float) -> np.ndarray:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax - vmin <= 0:
            return np.zeros((len(series), 1), dtype=float)
        normed = (values - vmin) / (vmax - vmin)
        # Missing values default to the column mean of the normed range (0.5).
        mask = np.isnan(normed)
        if mask.any():
            normed = np.where(mask, 0.5, normed)
        return (normed * np.sqrt(weight)).reshape(-1, 1)

    def _encode(self, df: pd.DataFrame, specs: List[FeatureSpec]) -> np.ndarray:
        blocks = []
        for col, kind, weight in specs:
            series = df[col]
            if kind == "nominal":
                block = self._encode_nominal(series, weight)
            elif kind == "ordinal":
                block = self._encode_ordinal(series, weight, self.ordinal_orders[col])
            else:
                block = self._encode_interval(series, weight)
            blocks.append(block)
        return np.hstack(blocks)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """Return the weighted encoded feature matrix without clustering.

        Useful when sweeping k or comparing weight schemes: encode once,
        then run sklearn.KMeans externally on the returned matrix. Populates
        ``self.dropped_features_`` and ``self.encoded_matrix_`` as a side
        effect.
        """
        specs = self._prepare_specs(df)
        X = self._encode(df.reset_index(drop=True), specs)
        self.encoded_matrix_ = X
        return X

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)
        df = df.copy().reset_index(drop=True)
        specs = self._prepare_specs(df)

        X = self._encode(df, specs)
        self.encoded_matrix_ = X

        kmeans = KMeans(
            n_clusters=self.n_partitions,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        labels = kmeans.fit_predict(X)
        self.kmeans_ = kmeans
        self.cluster_labels_ = labels

        partitions = []
        for i in range(self.n_partitions):
            part = df[labels == i].copy().reset_index(drop=True)
            part["_cluster_id"] = i
            partitions.append(part)
        return partitions
