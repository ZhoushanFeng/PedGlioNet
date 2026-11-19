#!/usr/bin/env python3
"""Redraw Figures 3-4 from audited, full-precision observed results.

Figure 3: paired forest plots, a zero-centred source/seed heatmap, and
reference-status contribution bars. Figure 4: paired per-seed dumbbells.
No estimates, confidence intervals, or hypothesis tests are recomputed.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/figures"
TRANSFER = ROOT / "output/adult_transfer_replication.json"
DECOMPOSITION = ROOT / "output/decomposition.json"
PREVIOUS = ROOT / "figure_data"
SEEDS = (2026, 2027, 2028)
REGIONS = ("WT", "TC", "ET", "NET", "CC", "ED")
POSITIVE = "#2F7F73"
ALL_CASES = "#515C69"
ABSENT = "#CAB073"
MODEL_COLORS = {"b0": "#285A80", "adult_init": "#BC783B"}
INK = "#29333C"
GRID = "#E6E9EB"


def load_source(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.relative_to(ROOT)),
                             "sha256": hashlib.sha256(raw).hexdigest()}


def configure():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9.5,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 10,
        "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .7,
        "axes.edgecolor": "#66717A", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "pdf.use14corefonts": False,
        "savefig.facecolor": "white",
    })


def label_panel(fig, ax, label):
    position = ax.get_position()
    fig.text(position.x0 - .070, position.y1 + .018, label,
             fontsize=12.5, fontweight="bold", va="bottom")


def format_axes(ax):
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3, pad=4)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=.55)
    ax.set_axisbelow(True)


def finish(fig, stem, metadata, caption):
    # Fixed canvas size preserves the exact manuscript width of 7.2 inches.
    fig.canvas.draw()
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=600)
    plt.close(fig)
    figure = stem.split("_")[0]
    (OUT / f"{figure}_data.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (OUT / f"{figure}_caption.txt").write_text(caption + "\n")
    print(f"Created {stem}: PDF, editable SVG, 600-dpi PNG.")


def figure3(adult, decomposition, source_metadata):
    old, old_meta = load_source(PREVIOUS / "Figure3.json")
    data = {"figure": "Figure3_transfer", "sources": source_metadata + [old_meta],
            "paired_effects": [], "source_seed": [], "decomposition": [],
            "uncertainty": "All 24 original paired case-bootstrap confidence intervals retained without recalculation.",
            "display": "Linear Dice differences in native 0-1 Dice units; reference line at zero."}
    fig = plt.figure(figsize=(7.2, 6.5))
    axes = [fig.add_axes([.095, .525, .365, .365]),
            fig.add_axes([.605, .525, .365, .365]),
            fig.add_axes([.095, .135, .300, .220]),
            fig.add_axes([.605, .135, .365, .220])]
    for ax, key, title in zip(axes[:2],
            ("three_seed_mean_case_dice", "new_seeds_mean_case_dice"),
            ("All 3 seeds", "New 2 seeds")):
        for index in range(6):
            if index % 2 == 0:
                ax.axhspan(index - .48, index + .48, color="#F4F6F6", zorder=0)
        for offset, subset, color, marker in ((-.16, "positive", POSITIVE, "o"),
                                              (.16, "all", ALL_CASES, "s")):
            selected = [next(r for r in adult[key]
                             if r["scope"] == "pooled" and r["region"] == region
                             and r["subset"] == subset) for region in REGIONS]
            means = np.array([r["paired_difference_adult_minus_b0"]["mean"] for r in selected])
            ci = np.array([r["paired_difference_adult_minus_b0"]["ci95"] for r in selected])
            assert ci.min() > -.05 and ci.max() < .07
            ax.errorbar(means, np.arange(6) + offset,
                        xerr=[means-ci[:, 0], ci[:, 1]-means],
                        fmt=marker, markersize=4.1, markeredgewidth=.8,
                        markerfacecolor=color if subset == "positive" else "white",
                        color=color, elinewidth=1.0, capsize=2.0, capthick=.8, zorder=3)
            data["paired_effects"].extend({"group": key, **r} for r in selected)
        ax.axvline(0, color="#79828A", lw=.9, ls=(0, (3, 3)), zorder=2)
        ax.set(yticks=np.arange(6), yticklabels=REGIONS, ylim=(5.6, -.6),
               xlim=(-.05, .07), xticks=[-.04, -.02, 0, .02, .04, .06],
               xlabel="Dice difference")
        ax.set_xticklabels(["-0.04", "-0.02", "0", "0.02", "0.04", "0.06"])
        ax.set_title(title, pad=11)
        format_axes(ax)
    fig.legend(handles=[
        Line2D([], [], color=POSITIVE, marker="o", markersize=4.3, lw=.9,
               label="Reference-positive"),
        Line2D([], [], color=ALL_CASES, marker="s", markersize=4.3,
               markerfacecolor="white", lw=.9, label="All cases")],
        loc="upper center", bbox_to_anchor=(.535, .995), ncol=2,
        frameon=False, columnspacing=2.8, handlelength=1.6)

    ax = axes[2]
    matrix = np.zeros((3, 3))
    for fold in range(3):
        for j, seed in enumerate(SEEDS):
            row = next(r for r in adult["per_seed_dice"] if r["region"] == "ET"
                       and r["scope"] == f"fold{fold}" and r["subset"] == "positive"
                       and r["seed"] == seed)
            matrix[fold, j] = row["paired_difference_adult_minus_b0"]["mean"]
            data["source_seed"].append(row)
    cmap = LinearSegmentedColormap.from_list("signed_delta", ["#B65555", "#FBFBFA", "#466BB0"])
    im = ax.imshow(matrix, norm=TwoSlopeNorm(vmin=-.065, vcenter=0, vmax=.065),
                   cmap=cmap, aspect="auto")
    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            ax.text(j, i, f"{value:+.3f}" if abs(value) >= .0005 else "0.000",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(value) > .039 else INK)
    ax.set(xticks=range(3), xticklabels=[str(s) for s in SEEDS],
           yticks=range(3), yticklabels=["CBTN", "DFCI", "DIPGr"], xlabel="Seed")
    ax.set_xticks(np.arange(4)-.5, minor=True)
    ax.set_yticks(np.arange(4)-.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("ET-positive by source", pad=13)
    cax = fig.add_axes([.418, .135, .014, .220])
    cb = fig.colorbar(im, cax=cax, ticks=[-.06, -.03, 0, .03, .06])
    cb.ax.set_yticklabels(["-0.06", "-0.03", "0", "0.03", "0.06"])
    cb.ax.tick_params(labelsize=8, length=2, pad=3)
    cb.outline.set_linewidth(.5)

    ax = axes[3]
    scopes = ["new_seeds_mean_all_sources", "new_seeds_mean_fold0_CBTN",
              "new_seeds_mean_fold1_DFCI-BCH-BWH-PEDs-HGG", "new_seeds_mean_fold2_DIPGr"]
    for i, scope in enumerate(scopes):
        row = next(r for r in decomposition["results"] if r["scope"] == scope and r["region"] == "ET")
        parts = row["dice_delta_decomposition"]
        positive = parts["positive_contribution"]["estimate"]
        negative = parts["negative_contribution"]["estimate"]
        net = row["dice_all"]["difference_adult_minus_b0"]
        assert np.isclose(positive + negative, net, atol=1e-12)
        starts = {True: 0., False: 0.}
        for value, color in ((positive, POSITIVE), (negative, ABSENT)):
            ax.barh(i, value, left=starts[value >= 0], height=.49,
                    color=color, edgecolor="white", linewidth=.35, zorder=3)
            starts[value >= 0] += value
        ax.plot(net, i, "D", color=INK, markersize=4, zorder=4)
        data["decomposition"].append(row)
    ax.axvline(0, color="#79828A", lw=.9, zorder=2)
    ax.set(yticks=range(4), yticklabels=["Pooled", "CBTN", "DFCI", "DIPGr"],
           ylim=(3.5, -.6), xlim=(-.014, .103), xticks=[0, .03, .06, .09],
           xlabel="ET Dice contribution")
    ax.set_xticklabels(["0", "0.03", "0.06", "0.09"])
    ax.set_title("Gain composition", pad=13)
    format_axes(ax)
    ax.legend(handles=[Patch(facecolor=POSITIVE, label="ET present"),
                       Patch(facecolor=ABSENT, label="ET absent"),
                       Line2D([], [], color=INK, marker="D", ls="", markersize=3.6, label="Net")],
              loc="upper center", bbox_to_anchor=(.50, -.36), ncol=3,
              frameon=False, fontsize=8.2, handlelength=.9,
              handletextpad=.4, columnspacing=.9, borderaxespad=0)
    for ax, label in zip(axes, "ABCD"):
        label_panel(fig, ax, label)
    assert len(data["paired_effects"]) == 24
    assert data["paired_effects"] == old["paired_effects"]
    assert data["source_seed"] == old["source_seed"]
    assert data["decomposition"] == old["decomposition"]
    data["validation"] = {"paired_intervals_retained": 24, "source_seed_values_retained": 9,
                          "decomposition_rows_retained": 4, "exact_match_to_previous_data": True,
                          "heatmap_range": [-.065, .065], "heatmap_center": 0}
    finish(fig, "Figure3_transfer", data,
        "Figure 3. Regional transfer gains and their source composition. "
        "(A,B) Forest plots of paired mean Dice differences (AdultInit minus B0) across all three pediatric "
        "seeds (2026-2028) and the two additional seeds (2027-2028), respectively. Teal circles denote "
        "reference-positive cases and open gray squares all cases. Horizontal intervals are the original "
        "endpoint-specific 95% source-stratified paired case-bootstrap intervals; values right of the zero "
        "line favor AdultInit. Per-case metrics are averaged over the indicated seeds before the paired "
        "case analysis. All-case n = 257 for every region; reference-positive n = 257, 257, 171, 256, 83 "
        "and 52 for WT, TC, ET, NET, CC and ED, respectively. "
        "(C) ET-positive Dice differences by source and pediatric seed. The red-to-blue scale is centered "
        "at zero and spans -0.065 to +0.065; blue denotes positive gain and red negative gain. Cell labels "
        "are rounded to three decimals. Source-specific ET-positive counts are 87 CBTN, 35 DFCI and 49 DIPGr. "
        "(D) Additional-seed all-case ET gains decomposed into reference-positive (ET-present, teal) and "
        "reference-negative (ET-absent, gold) contributions within each population. Signed segments start "
        "at zero and stack in their respective direction; black diamonds mark the algebraic net sum. "
        "Denominators are 257 pooled, 116 CBTN, 61 DFCI and 80 DIPGr cases. Segments and diamonds are "
        "descriptive estimates; no new intervals or significance tests were added. DFCI abbreviates the "
        "DFCI-BCH-BWH-PEDs-HGG source. Dice differences use native 0-1 Dice units throughout.")


def figure4(decomposition, source_metadata):
    old, old_meta = load_source(PREVIOUS / "Figure5.json")
    records = []
    for seed in SEEDS:
        selected = [r for r in decomposition["results"] if r["region"] == "ET"
                    and r["scope"] == f"seed{seed}_all_sources"]
        assert len(selected) == 1
        record = selected[0]
        assert (record["n_reference_positive"], record["n_reference_negative"],
                record["n_unique_cases"]) == (171, 86, 257)
        records.append(record)
    panels = [
        {"letter": "A", "title": "ET omissions", "parent": "reference_positive_audit",
         "key": "complete_empty_predictions", "xlabel": "Cases", "limit": 35,
         "ticks": [0, 10, 20, 30], "format": ".0f", "n_cases": 171},
        {"letter": "B", "title": "ET false positives", "parent": "reference_negative_audit",
         "key": "false_positive_cases", "xlabel": "Cases", "limit": 25,
         "ticks": [0, 5, 10, 15, 20, 25], "format": ".0f", "n_cases": 86},
        {"letter": "C", "title": "False-positive volume", "parent": "reference_negative_audit",
         "key": "false_positive_volume_mean_per_negative_case_mm3", "xlabel": "Mean volume (mm³)",
         "limit": 180, "ticks": [0, 50, 100, 150], "format": ".1f", "n_cases": 86},
        {"letter": "D", "title": "Component precision", "parent": "connected_component_audit",
         "key": "ppv", "xlabel": "PPV", "limit": 1,
         "ticks": [0, .25, .5, .75, 1], "format": ".3f", "n_cases": 257},
    ]
    fig = plt.figure(figsize=(7.2, 5.35))
    axes = [fig.add_axes([.100, .580, .355, .275]),
            fig.add_axes([.610, .580, .355, .275]),
            fig.add_axes([.100, .145, .355, .275]),
            fig.add_axes([.610, .145, .355, .275])]
    exported = []
    for ax, panel in zip(axes, panels):
        values = [r[panel["parent"]][panel["key"]] for r in records]
        panel_rows = []
        for i, (seed, value) in enumerate(zip(SEEDS, values)):
            b0, adult = value["b0"], value["adult_init"]
            ax.plot([b0, adult], [i, i], color="#AAB2B8", lw=1.4, zorder=2)
            # The larger open B0 circle remains visible when both values coincide.
            ax.plot(b0, i, "o", ms=6.4, markerfacecolor="white",
                    markeredgecolor=MODEL_COLORS["b0"], markeredgewidth=1.15, zorder=3)
            ax.plot(adult, i, "o", ms=4.1, markerfacecolor=MODEL_COLORS["adult_init"],
                    markeredgecolor=MODEL_COLORS["adult_init"], markeredgewidth=.6, zorder=4)
            ax.annotate(format(b0, panel["format"]), (b0, i), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom",
                        color=MODEL_COLORS["b0"], fontsize=8.7)
            ax.annotate(format(adult, panel["format"]), (adult, i), xytext=(0, -9),
                        textcoords="offset points", ha="center", va="top",
                        color=MODEL_COLORS["adult_init"], fontsize=8.7)
            panel_rows.append({"seed": seed, "b0": b0, "adult_init": adult, "unit": value["unit"]})
        ax.set(xlim=(0, panel["limit"]), ylim=(2.50, -.52),
               yticks=range(3), yticklabels=[str(s) for s in SEEDS],
               xticks=panel["ticks"], xlabel=panel["xlabel"])
        if panel["letter"] == "D":
            ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
        ax.set_title(panel["title"], pad=16)
        format_axes(ax)
        label_panel(fig, ax, panel["letter"])
        exported.append({"panel": panel["letter"], "metric": panel["key"],
                         "n_cases": panel["n_cases"], "values": panel_rows})
    fig.legend(handles=[
        Line2D([], [], marker="o", ms=6.4, markerfacecolor="white",
               markeredgecolor=MODEL_COLORS["b0"], markeredgewidth=1.15, ls="", label="B0"),
        Line2D([], [], marker="o", ms=4.1, color=MODEL_COLORS["adult_init"], ls="", label="AdultInit")],
        loc="upper center", bbox_to_anchor=(.535, .998), ncol=2, frameon=False,
        handlelength=1, columnspacing=2.5, fontsize=10)
    assert exported == old["panels"]
    data = {"figure": "Figure5_et_errors", "sources": [source_metadata, old_meta],
            "panels": exported, "uncertainty": "Observed point estimates only; no confidence intervals or significance annotations added.",
            "display": "Each line joins B0 and AdultInit for the same seed and evaluation cases. Labels are rounded only for display.",
            "zero_baselines": True, "ppv_axis_maximum": 1,
            "validation": {"exact_match_to_previous_data": True, "observations_retained": 24,
                           "same_value_nested_markers": True}}
    finish(fig, "Figure5_et_errors", data,
        "Figure 5. ET outcomes across three paired training seeds. Each horizontal gray line joins B0 "
        "(open blue circle) and AdultInit (filled orange circle) for the same seed; nested circles indicate "
        "equal values. Numeric labels above and below each line give B0 and AdultInit values, respectively. "
        "(A) Complete ET omissions among 171 reference-positive cases. "
        "(B) Cases with any false-positive ET prediction among 86 reference-negative cases. "
        "(C) Mean false-positive ET volume per reference-negative case, including zero volumes (n = 86). "
        "(D) Predicted-component positive predictive value (PPV), the number of predicted ET components "
        "with any reference overlap divided by all predicted ET components across 257 cases. "
        "Lower values are preferable in A-C; higher values are preferable in D. Each seed evaluates the "
        "same cases. Counts, volumes and PPV are observed point estimates; no intervals or significance "
        "annotations are added. Paired-difference intervals and source-level results are provided in "
        "Supplementary Table S3.")


def main():
    configure()
    adult, adult_meta = load_source(TRANSFER)
    decomposition, decomposition_meta = load_source(DECOMPOSITION)
    figure3(adult, decomposition, [adult_meta, decomposition_meta])
    figure4(decomposition, decomposition_meta)


if __name__ == "__main__":
    main()
