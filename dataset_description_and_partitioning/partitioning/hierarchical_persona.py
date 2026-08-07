from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import DatasetPartitioner


class HierarchicalPersonaPartitioner(DatasetPartitioner):
    """P1 — Hierarchical deterministic (design x persona bin(s)) partitioner.

    Each client is one (design_variant, performance_persona) team. Cells are
    the Cartesian product of ``design_col`` values x quantile bins of one or
    more numeric persona columns. Every sample lands in exactly one cell;
    the cell id becomes the client id.

    ``persona_col`` accepts a single string (single persona axis, e.g.
    ``clock_ns``) or a list of strings (multi-axis, e.g.
    ``['clock_ns', 'utilization']``). ``n_persona_bins`` mirrors that:
    a single int applies to every axis, or a list gives per-axis bin counts.

    ``n_partitions`` must equal ``n_unique(design_col) *
    prod(n_persona_bins_per_axis)``. If they mismatch, the ValueError message
    reports the natural cell count so you can adjust the binning. Set
    ``n_persona_bins=1`` (single-axis) to recover a pure leave-one-design-out
    split; on any given axis a bin count of 1 skips that axis.

    Args:
        n_partitions: Number of federated parties.
        design_col: Categorical column defining the client family.
        persona_col: Numeric column, list of numeric columns, or None. When
            None (or an empty list), only the design axis is used.
        n_persona_bins: Number of persona bins per axis. Either an int
            broadcast to every axis, or a list matching ``persona_col``.
        persona_bin_edges: Optional explicit bin edges. For a single axis:
            a flat sequence of floats. For multiple axes: a list of edge
            sequences (or ``None`` per axis to fall back on quantile /
            equal_width binning).
        bin_method: ``"quantile"`` or ``"equal_width"``. Ignored on any axis
            with explicit ``persona_bin_edges``.
        seed: Kept for interface parity; output is deterministic given the
            input DataFrame.
    """

    def __init__(
        self,
        n_partitions: int,
        design_col: str = "design_name",
        persona_col: Optional[Union[str, Sequence[str]]] = "clock_ns",
        n_persona_bins: Union[int, Sequence[int]] = 2,
        persona_bin_edges: Optional[Sequence] = None,
        bin_method: str = "quantile",
        seed: int = 42,
    ) -> None:
        super().__init__(n_partitions)
        if bin_method not in ("quantile", "equal_width"):
            raise ValueError("bin_method must be 'quantile' or 'equal_width'")

        # Normalize persona_col to a list of column names.
        if persona_col is None:
            self._persona_cols: List[str] = []
        elif isinstance(persona_col, str):
            self._persona_cols = [persona_col]
        else:
            self._persona_cols = list(persona_col)
        n_axes = len(self._persona_cols)

        # Normalize n_persona_bins to a list matching persona_cols.
        if isinstance(n_persona_bins, int):
            self._n_bins_list: List[int] = [n_persona_bins] * n_axes
        else:
            self._n_bins_list = list(n_persona_bins)
            if len(self._n_bins_list) != n_axes:
                raise ValueError(
                    f"n_persona_bins has {len(self._n_bins_list)} entries but "
                    f"persona_col has {n_axes} axes"
                )
        for nb in self._n_bins_list:
            if nb < 1:
                raise ValueError("every n_persona_bins entry must be >= 1")
        if any(nb > 1 for nb in self._n_bins_list) and n_axes == 0:
            raise ValueError("persona_col is required when any n_persona_bins > 1")

        # Normalize persona_bin_edges to a list of (edge-seq or None) per axis.
        if persona_bin_edges is None:
            self._edges_list: List[Optional[List[float]]] = [None] * n_axes
        elif n_axes <= 1:
            # Single-axis backward compat: a flat sequence of floats.
            self._edges_list = [list(persona_bin_edges)] if n_axes == 1 else []
        else:
            # Multi-axis: a list of edge sequences (or None per axis).
            edges_seq = list(persona_bin_edges)
            if len(edges_seq) != n_axes:
                raise ValueError(
                    f"persona_bin_edges has {len(edges_seq)} entries but "
                    f"persona_col has {n_axes} axes"
                )
            self._edges_list = [
                (list(e) if e is not None else None) for e in edges_seq
            ]

        self.design_col = design_col
        self.persona_col = persona_col
        self.n_persona_bins = n_persona_bins
        self.persona_bin_edges = persona_bin_edges
        self.bin_method = bin_method
        self.seed = seed

    def _compute_bin_edges(
        self,
        values: np.ndarray,
        n_bins: int,
        explicit_edges: Optional[Sequence[float]],
        col_name: str,
    ) -> np.ndarray:
        if explicit_edges is not None:
            edges = np.asarray(explicit_edges, dtype=float)
        elif self.bin_method == "quantile":
            edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        else:
            edges = np.linspace(values.min(), values.max(), n_bins + 1)
        # Widen the outer edges slightly so min/max fall inside the bin range.
        edges = edges.astype(float).copy()
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        # Deduplicate (quantile edges can collapse when a value dominates).
        edges = np.unique(edges)
        if len(edges) - 1 != n_bins:
            raise ValueError(
                f"Could not build {n_bins} distinct persona bins from "
                f"persona_col='{col_name}' (values collapse to "
                f"{len(edges) - 1} bins). Reduce n_persona_bins or provide "
                "persona_bin_edges explicitly."
            )
        return edges

    def _cell_labels(self, df: pd.DataFrame) -> pd.Series:
        if self.design_col not in df.columns:
            raise ValueError(f"design_col '{self.design_col}' not found in DataFrame")
        label = df[self.design_col].astype(str)

        for col, n_bins, explicit_edges in zip(
            self._persona_cols, self._n_bins_list, self._edges_list
        ):
            if n_bins == 1:
                # Single degenerate bin => this axis contributes nothing.
                continue
            if col not in df.columns:
                raise ValueError(f"persona_col '{col}' not found in DataFrame")
            values = pd.to_numeric(df[col], errors="raise").to_numpy()
            edges = self._compute_bin_edges(values, n_bins, explicit_edges, col)
            bin_idx = np.searchsorted(edges[1:], values, side="left")
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            suffix = pd.Series(bin_idx, index=df.index).map(
                lambda b, c=col: f"{c}{b}"
            )
            label = label + "|" + suffix
        return label

    def partition(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        self._validate(df)
        df = df.copy().reset_index(drop=True)

        cell_key = self._cell_labels(df)
        unique_cells = sorted(cell_key.unique().tolist())
        n_cells = len(unique_cells)

        if n_cells != self.n_partitions:
            axis_desc = " x ".join(
                [self.design_col]
                + [
                    f"{c}_bin({nb})"
                    for c, nb in zip(self._persona_cols, self._n_bins_list)
                    if nb > 1
                ]
            )
            raise ValueError(
                f"HierarchicalPersonaPartitioner: n_partitions={self.n_partitions} "
                f"but the natural ({axis_desc}) grid produced {n_cells} cells. "
                f"Adjust n_persona_bins so that the product matches n_partitions "
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
