import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class LabelTierAssigner:
    """Categorizes CircuitNet-N28 labels into four violation-severity tiers.

    A pixel is a DRC violation if its normalized value >= ``threshold``.
    The per-label *violation rate* (fraction of violating pixels) is then
    mapped to one of four ordinal tiers.

    Default tier boundaries (violation rate):
        Tier 0  clean   [0.00, 0.02)  — nearly no violations
        Tier 1  low     [0.02, 0.08)  — sparse violations
        Tier 2  medium  [0.08, 0.20)  — moderate violations
        Tier 3  high    [0.20, 1.00]  — dense violations

    Args:
        threshold:  Pixel-level binarization threshold (default: 0.1).
        boundaries: Strictly-increasing violation-rate breakpoints defining
            the left edge of each tier plus a sentinel value > 1.
            Must have exactly ``n_tiers + 1`` elements.
    """

    TIER_NAMES: Dict[int, str] = {0: "clean", 1: "low", 2: "medium", 3: "high"}
    _DEFAULT_BOUNDARIES: List[float] = [0.0, 0.02, 0.08, 0.20, 1.01]

    def __init__(
        self,
        threshold: float = 0.1,
        boundaries: Optional[List[float]] = None,
    ) -> None:
        self.threshold = threshold
        self.boundaries = boundaries or self._DEFAULT_BOUNDARIES
        self._validate_boundaries()

    def _validate_boundaries(self) -> None:
        b = self.boundaries
        if len(b) < 2:
            raise ValueError("boundaries must have at least 2 values.")
        if b[0] != 0.0:
            raise ValueError("boundaries must start at 0.0.")
        if not all(b[i] < b[i + 1] for i in range(len(b) - 1)):
            raise ValueError("boundaries must be strictly increasing.")

    @property
    def n_tiers(self) -> int:
        return len(self.boundaries) - 1

    def violation_rate(self, label: np.ndarray) -> float:
        """Fraction of pixels in *label* with value >= :attr:`threshold`."""
        return float(np.mean(label >= self.threshold))

    def assign_tier(self, rate: float) -> int:
        """Map a violation *rate* to its tier index [0 .. n_tiers-1]."""
        for i in range(self.n_tiers):
            if rate < self.boundaries[i + 1]:
                return i
        return self.n_tiers - 1

    def label_stats(self, label: np.ndarray) -> Dict:
        """Compute per-label summary statistics."""
        flat = label.ravel()
        rate = float(np.mean(flat >= self.threshold))
        tier = self.assign_tier(rate)
        return {
            "violation_count": int(np.sum(flat >= self.threshold)),
            "violation_rate": rate,
            "tier": tier,
            "tier_name": self.TIER_NAMES.get(tier, "unknown"),
            "label_mean": float(np.mean(flat)),
            "label_max": float(np.max(flat)),
        }

    def process_csv(self, csv_path: str, dataroot: str) -> pd.DataFrame:
        """Scan every label in *csv_path* and return a per-sample stats DataFrame.

        The CSV must follow the CircuitNet annotation convention:
            feature/<name>.npy,label/<name>.npy

        Returns a DataFrame with columns:
            feature_path, label_path, violation_count, violation_rate,
            tier, tier_name, label_mean, label_max.
        """
        rows = []
        with open(csv_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                feature_rel, label_rel = line.split(",")
                label_path = os.path.join(dataroot, label_rel.strip())
                label = np.load(label_path)
                stats = self.label_stats(label)
                rows.append({"feature_path": feature_rel.strip(), "label_path": label_rel.strip(), **stats})
        return pd.DataFrame(rows)

    def tier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a per-tier descriptive statistics table from a processed DataFrame."""
        groups = df.groupby("tier").agg(
            count=("violation_rate", "size"),
            mean_rate=("violation_rate", "mean"),
            median_rate=("violation_rate", "median"),
            mean_count=("violation_count", "mean"),
            max_count=("violation_count", "max"),
        )
        groups.index = groups.index.map(lambda t: f"{t} ({self.TIER_NAMES.get(t, '?')})")
        return groups

    def plot_distribution(self, df: pd.DataFrame) -> None:
        """Plot tier bar chart and violation-rate histogram side by side."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        tier_counts = df["tier"].value_counts().sort_index().rename(index=self.TIER_NAMES)
        colors = ["#4caf50", "#ff9800", "#f44336", "#9c27b0"]
        tier_counts.plot.bar(ax=axes[0], color=colors[: len(tier_counts)])
        axes[0].set_title("DRC violation severity — tier distribution")
        axes[0].set_xlabel("Tier")
        axes[0].set_ylabel("Sample count")
        axes[0].tick_params(axis="x", rotation=0)
        for bar, cnt in zip(axes[0].patches, tier_counts):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                str(cnt),
                ha="center",
                fontsize=9,
            )

        axes[1].hist(df["violation_rate"], bins=50, color="#2196f3", alpha=0.75, edgecolor="white")
        for boundary in self.boundaries[1:-1]:
            axes[1].axvline(boundary, color="red", linestyle="--", linewidth=1.2,
                            label=f"boundary {boundary}")
        axes[1].set_title(f"Violation rate distribution  (threshold = {self.threshold})")
        axes[1].set_xlabel("Violation rate (fraction of pixels ≥ threshold)")
        axes[1].set_ylabel("Sample count")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.show()
