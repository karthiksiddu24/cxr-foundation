"""Shared matplotlib style for every figure in docs/ and RESULTS.md.

Colors come from a CVD-validated categorical palette, assigned in fixed slot order (never cycled).
Charts with >3 simultaneously-compared series must facet or fold rather than add hues.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#8a8984"  # reference lines / baselines

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3 (validated all-pairs)

# Fixed identity -> color, so an init keeps its color in every figure.
INIT_COLORS = {"random": SERIES[1], "imagenet": SERIES[0], "ssl": SERIES[2]}
SPLIT_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.family": "DejaVu Sans",
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
