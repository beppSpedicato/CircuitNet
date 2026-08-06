from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class HierarchicalPersonaPartitioner(DatasetPartitioner):
    """P1 — Hierarchical deterministic (design x persona bin) partitioner.

    Each client is one (design_variant, performance_persona) team. Cells are
    the Cartesian product of ``design_col`` values x quantile bins of a
    numeric ``persona_col`` (typically ``clock_ns`` on N28 or ``freq_mhz``
    on N14). Every sample lands in exactly one cell; the cell id becomes
    the client id.

    ``n_partitions`` must equal ``n_unique(design_col) * n_persona_bins``.
    If they mismatch, the ValueError message reports the natural cell count
    so you can adjust ``n_persona_bins`` (or preprocess ``design_col``).
    Set ``n_persona_bins=1`` to recover a pure leave-one-design-out split.

    Args:
        n_partitions: Number of federated parties.
        design_col: Categorical column defining the client family.
        persona_col: Numeric column binned into personas. Set to None when
            ``n_persona_bins == 1``.
        n_persona_bins: Number of persona bins over persona_col (>= 1).
        persona_bin_edges: Optional explicit edges (overrides
            n_persona_bins-driven quantile edges).
        bin_method: ``"quantile"`` or ``"equal_width"``. Ignored when
            ``persona_bin_edges`` is provided.
        seed: Kept for interface parity; output is deterministic given
            the input DataFrame.
    """

    def __init__(
        self,
        n_partitions: int,
        design_col: str = "design_name",
        persona_col: Optional[str] = "clock_ns",
        n_persona_bins: int = 2,
        persona_bin_edges: Optional[Sequence[float]] = None,
        bin_method: str = "quantile",
        seed: int = 42,
    ) -> None:
        super().__init__(n_partitions)
        if n_persona_bins < 1:
            raise ValueError("n_persona_bins must be >= 1")
        if bin_method not in ("quantile", "equal_width"):
            raise ValueError("bin_method must be 'quantile' or 'equal_width'")
        if n_persona_bins > 1 and persona_col is None:
            raise ValueError("persona_col is required when n_persona_bins > 1")
        self.design_col = design_col
        self.persona_col = persona_col
        self.n_persona_bins = n_persona_bins
        self.persona_bin_edges = (
            list(persona_bin_edges) if persona_bin_edges is not None else None
        )
        self.bin_method = bin_method
        self.seed = seed

    def _compute_bin_edges(self, values: np.ndarray) -> np.ndarray:
        if self.persona_bin_edges is not None:
            edges = np.asarray(self.persona_bin_edges, dtype=float)
        elif self.bin_method == "quantile":
            edges = np.quantile(values, np.linspace(0, 1, self.n_persona_bins + 1))
        else:
            edges = np.linspace(values.min(), values.max(), self.n_persona_bins + 1)
        # Widen the outer edges slightly so min/max fall inside the bin range.
        edges = edges.astype(float).copy()
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        # Deduplicate (quantile edges can collapse when a value dominates).
        edges = np.unique(edges)
        if len(edges) - 1 != self.n_persona_bins:
            raise ValueError(
                f"Could not build {self.n_persona_bins} distinct persona bins from "
                f"persona_col='{self.persona_col}' (values collapse to "
                f"{len(edges) - 1} bins). Reduce n_persona_bins or provide "
                "persona_bin_edges explicitly."
            )
        return edges

    def _cell_labels(self, df: pd.DataFrame) -> pd.Series:
        if self.design_col not in df.columns:
            raise ValueError(f"design_col '{self.design_col}' not found in DataFrame")
        design = df[self.design_col].astype(str)

        if self.n_persona_bins == 1:
            return design

        if self.persona_col not in df.columns:
            raise ValueError(f"persona_col '{self.persona_col}' not found in DataFrame")
        values = pd.to_numeric(df[self.persona_col], errors="raise").to_numpy()
        edges = self._compute_bin_edges(values)
        bin_idx = np.searchsorted(edges[1:], values, side="left")
        bin_idx = np.clip(bin_idx, 0, self.n_persona_bins - 1)
        return design + "|" + pd.Series(bin_idx, index=df.index).map(lambda b: f"p{b}")

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)
        df = df.copy().reset_index(drop=True)

        cell_key = self._cell_labels(df)
        unique_cells = sorted(cell_key.unique().tolist())
        n_cells = len(unique_cells)

        if n_cells != self.n_partitions:
            raise ValueError(
                f"HierarchicalPersonaPartitioner: n_partitions={self.n_partitions} "
                f"but the natural (design x persona_bin) grid produced {n_cells} "
                f"cells. Adjust n_persona_bins so that "
                f"n_unique({self.design_col}) * n_persona_bins == n_partitions "
                f"(observed cells: {unique_cells})."
            )

        cell_to_party = {cell: i for i, cell in enumerate(unique_cells)}
        assignment = cell_key.map(cell_to_party).to_numpy()

        partitions = []
        for i in range(self.n_partitions):
            part = df[assignment == i].copy().reset_index(drop=True)
            part["_client_cell"] = unique_cells[i]
            partitions.append(part)
        return partitions
