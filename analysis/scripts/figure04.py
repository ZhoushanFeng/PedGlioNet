#!/usr/bin/env python3
"""Render the complete source-by-seed adult-transfer comparison as Figure 4."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/adult_transfer_replication.json"
DEST = ROOT / "output/figures"
STEM = "Figure4_source_seed"
REGIONS = ["ET", "NET", "CC", "ED", "TC", "WT"]
SOURCES = ["CBTN", "DFCI-BCH-BWH-PEDs-HGG", "DIPGr"]
SEEDS = [2026, 2027, 2028]
SUBSETS = [("all", "All cases"), ("positive", "Reference-positive")]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 9,
        "text.color": "#19242D",
        "axes.labelcolor": "#19242D",
        "xtick.color": "#19242D",
        "ytick.color": "#19242D",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "axes.unicode_minus": True,
    })

    raw = json.loads(SOURCE.read_text())
    input_rows = [r for r in raw["per_seed_dice"] if r["source"] in SOURCES]
    assert len(input_rows) == 108
    lookup = {}
    for row in input_rows:
        key = (row["subset"], row["region"], row["source"], row["seed"])
        assert key not in lookup, f"Duplicate result: {key}"
        lookup[key] = row
    panels, data_rows = [], []
    for subset, _ in SUBSETS:
        matrix = np.empty((len(REGIONS), len(SOURCES) * len(SEEDS)))
        for i, region in enumerate(REGIONS):
            for k, source in enumerate(SOURCES):
                for j, seed in enumerate(SEEDS):
                    row = lookup[(subset, region, source, seed)]
                    difference = float(row["paired_difference_adult_minus_b0"]["mean"])
                    b0 = float(row["b0_dice"]["mean"])
                    adult = float(row["adult_init_dice"]["mean"])
                    assert math.isclose(adult - b0, difference, abs_tol=2e-15)
                    delta_pp = difference * 100.0
                    matrix[i, 3 * k + j] = delta_pp
                    data_rows.append({
                        "subset": subset, "region": region, "source": source,
                        "seed": seed, "n_cases": row["n_cases"],
                        "b0_dice": b0, "adult_init_dice": adult,
                        "difference_dice": difference, "difference_percentage_points": delta_pp,
                        "paired_ci95_dice": row["paired_difference_adult_minus_b0"]["ci95"],
                    })
        panels.append(matrix)
    all_values = np.stack(panels)
    assert all_values.shape == (2, 6, 9)
    assert np.isfinite(all_values).all()
    limit = math.ceil(float(np.abs(all_values).max()))
    assert limit == 14

    cmap = LinearSegmentedColormap.from_list("transfer_direction", ["#B86A36", "#FCFCFB", "#21618A"], N=257)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig = plt.figure(figsize=(7.12, 3.35), layout=None)
    axes = [fig.add_axes([0.069, 0.30, 0.400, 0.485]),
            fig.add_axes([0.572, 0.30, 0.400, 0.485])]

    for p, (ax, matrix, (_, title)) in enumerate(zip(axes, panels, SUBSETS)):
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(9), labels=[str(seed) for _ in SOURCES for seed in SEEDS])
        ax.set_yticks(np.arange(6), labels=REGIONS)
        ax.tick_params(axis="both", which="both", length=0, pad=4)
        ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        for boundary in (2.5, 5.5):
            ax.axvline(boundary, color="white", linewidth=2.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for x, label in zip((1, 4, 7), ("CBTN", "DFCI", "DIPGr")):
            ax.text(x, 1.09, label, ha="center", va="center", fontsize=9,
                    transform=ax.get_xaxis_transform())
        ax.text(-0.085, 1.34, chr(65 + p), transform=ax.transAxes,
                ha="left", va="center", fontsize=12, fontweight="bold")
        ax.text(0.0, 1.34, title, transform=ax.transAxes,
                ha="left", va="center", fontsize=11)
        for i in range(6):
            for j in range(9):
                value = float(matrix[i, j])
                # Avoid displaying negative zero after rounding.
                label = "0.0" if round(value, 1) == 0 else f"{value:+.1f}"
                text_color = "white" if abs(value) > 9.5 else "#19242D"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.7, color=text_color)

    cax = fig.add_axes([0.34, 0.133, 0.36, 0.032])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-14, -7, 0, 7, 14])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, width=0.5, pad=3, labelsize=8)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x:+.0f}"))
    cbar.set_label("ΔDice (percentage points)", labelpad=3, fontsize=9)

    pdf = DEST / f"{STEM}.pdf"
    fig.savefig(pdf, metadata={"Title": "Source and seed variation in adult-transfer effects", "Author": "PedGlioNet"})
    fig.savefig(DEST / f"{STEM}.svg")
    fig.savefig(DEST / f"{STEM}.png", dpi=600)
    plt.close(fig)

    caption = (
        "Figure 4. Adult-transfer effects across sources and training seeds. "
        "Each cell shows the paired mean Dice difference (AdultInit minus pediatric-only B0), "
        "in percentage points, for all cases (A) or cases with the indicated reference region present (B). "
        "Both panels use the same zero-centered color scale: blue denotes higher Dice with AdultInit "
        "and orange denotes lower Dice. Columns are grouped by held-out source; seeds are 2026, 2027, and 2028. "
        "DFCI abbreviates DFCI-BCH-BWH-PEDs-HGG. ET, enhancing tumor; NET, non-enhancing tumor; "
        "CC, cystic component; ED, edema; TC, tumor core; WT, whole tumor. "
        "Displayed numbers are point estimates rounded to one decimal place.\n"
    )
    (DEST / "Figure4_caption.txt").write_text(caption)
    audit = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_array": "per_seed_dice",
        "font": "DejaVu Serif",
        "font_file": "Matplotlib bundled DejaVu Serif",
        "rows": REGIONS, "sources": SOURCES, "seeds": SEEDS,
        "panels": [subset for subset, _ in SUBSETS],
        "units": "percentage points: 100 * (AdultInit Dice - B0 Dice)",
        "colormap": {"negative": "#B86A36", "zero": "#FCFCFB", "positive": "#21618A"},
        "color_limits": [-limit, limit],
        "actual_minimum": float(all_values.min()), "actual_maximum": float(all_values.max()),
        "n_cells": len(data_rows), "n_negative": int((all_values < 0).sum()),
        "n_positive": int((all_values > 0).sum()),
        "records": data_rows,
    }
    (DEST / "Figure4_data.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("font_file", "color_limits", "actual_minimum", "actual_maximum", "n_cells", "n_negative", "n_positive")}, indent=2))
    print(pdf.name)


if __name__ == "__main__":
    main()
