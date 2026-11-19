from pathlib import Path
import hashlib
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from reliability import ordered_expected_losses

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 10,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "axes.linewidth": .65,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "pdf.fonttype": 42,
    "ps.fonttype": 42, "pdf.use14corefonts": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
BLUE, ORANGE, TEAL, RED = "#285A80", "#C37A30", "#32857B", "#A84841"
INPUTS = set()
FIGURE_DATA = {}
FIGURE_CAPTIONS = {}


def read_json(relative):
    INPUTS.add(relative)
    return json.loads((ROOT / relative).read_text())


def read_csv(relative):
    INPUTS.add(str(relative))
    return pd.read_csv(ROOT / relative)


def letter(ax, label, x=-.11):
    ax.text(x, 1.055, label, transform=ax.transAxes, fontweight="bold", fontsize=12,
            va="bottom", ha="left")


def format_ax(ax):
    ax.tick_params(width=.6, length=3)
    ax.set_axisbelow(True)


def save(fig, name):
    for ax, label in zip(fig.axes[:2], ['WT', 'ET-positive']):
        ax.text(.5, 1.045, label, ha='center', transform=ax.transAxes, fontsize=11)
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', pad_inches=.09)
    for ax in fig.axes:
        format_ax(ax)
        assert not ax.get_title(), "Explanatory figure titles belong in external captions."
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.09,
                metadata={"Title": name, "Creator": "PedGlioNet audited figure builder"})
    fig.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight", pad_inches=.09)
    plt.close(fig)


def roc(y, score):
    order = np.argsort(-score, kind="stable")
    s, y = score[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(y)-1]
    tp = np.cumsum(y)[ends]
    fp = ends+1-tp
    return np.r_[0, fp/(~y).sum()], np.r_[0, tp/y.sum()]


def quality():
    cases = pd.concat([read_csv(f"data/eval/ensemble_b0_fold{f}_per_case.csv") for f in range(3)])
    unc = pd.concat([read_csv(f"data/uncertainty/b0_fold{f}_uncertainty_per_case.csv") for f in range(3)])
    verified = read_csv("figure_data/reliability_discrimination.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.65))
    fig.subplots_adjust(left=.1, right=.985, bottom=.14, top=.9, wspace=.28, hspace=.42)
    recorded = []
    for col, (region, positive, target) in enumerate([("WT", False, "WT"), ("ET", True, "ET_positive")]):
        g = cases.loc[cases.region.eq(region)]
        if positive:
            g = g.loc[g.reference_nonempty]
        g = g.merge(unc, on=["case_id", "fold"], validate="one_to_one")
        dice = g.dice.to_numpy()
        y = dice < .5
        assert len(g) == (171 if positive else 257)
        assert y.sum() == (49 if positive else 12)
        signals = [("Disagreement", f"{region.lower()}_disagreement", 1-g[f"{region.lower()}_pairwise_dice"].to_numpy(), ORANGE),
                   ("Entropy", "mean_entropy", g.mean_entropy.to_numpy(), BLUE),
                   ("Low volume", f"low_predicted_{region.lower()}_volume", -np.log1p(g.prediction_volume_mm3.to_numpy()), TEAL)]
        for label, key, score, color in signals:
            fpr, tpr = roc(y, score)
            # Match the audited AUROC exactly before using these observed curves.
            auroc = np.trapezoid(tpr, fpr)
            ref = verified.loc[verified.target.eq(target) & verified.scope.eq("pooled") &
                               verified.signal.eq(key) & verified.failure_threshold.eq(.5)].iloc[0]
            assert np.isclose(auroc, ref.auroc_value, atol=1e-12), (target, key, auroc, ref.auroc_value)
            axes[0, col].plot(fpr, tpr, color=color, lw=1.5, label=label)
            loss = ordered_expected_losses(score, 1-dice)[0]
            risk = np.cumsum(loss)/np.arange(1, len(loss)+1)
            coverage = np.arange(1, len(loss)+1)/len(loss)
            axes[1, col].plot(coverage, risk, color=color, lw=1.5)
            assert np.isclose(risk.mean(), ref.aurc_value, atol=1e-12)
            recorded.append({"target": target, "signal": key, "n": len(g), "failures": int(y.sum()),
                             "auroc": float(auroc), "aurc": float(risk.mean()),
                             "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                             "coverage": coverage.tolist(), "risk": risk.tolist()})
        axes[0, col].plot([0, 1], [0, 1], ls="--", color="#9A9FA4", lw=.8)
        axes[0, col].set(xlim=(0, 1), ylim=(0, 1.015), xlabel="False-positive rate", ylabel="True-positive rate")
        axes[1, col].axhline((1-dice).mean(), color="#777E84", ls="--", lw=1)
        oracle = np.sort(1-dice)
        axes[1, col].plot(coverage, np.cumsum(oracle)/np.arange(1, len(oracle)+1),
                          color="#777E84", ls=":", lw=1.2)
        axes[1, col].set(xlim=(0, 1), ylim=(0, 1 if positive else .25),
                         xlabel="Retained fraction", ylabel="Mean error (1 - Dice)")
        for row in range(2):
            axes[row, col].set_xticks([0, .2, .4, .6, .8, 1])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(.55, 1), frameon=False, handlelength=2.1, columnspacing=2)
    fig.legend(handles=[Line2D([], [], ls="--", color="#777E84", label="Random"),
                        Line2D([], [], ls=":", color="#777E84", label="Oracle")],
               loc="lower center", ncol=2, bbox_to_anchor=(.55, 0), frameon=False,
               handlelength=2.1, columnspacing=2)
    for ax, lab in zip(axes.flat, "ABCD"):
        letter(ax, lab, x=-.15)
    name = "Figure6_quality"
    FIGURE_DATA[name] = recorded
    FIGURE_CAPTIONS[name] = (
        "Figure 6. Region-specific quality ranking of the historical B0 probability ensemble. "
        "(A, C) WT across 257 cases, including 12 failures; (B, D) ET among 171 reference-positive "
        "cases, including 49 failures. Failure is Dice <0.5. (A, B) Empirical receiver operating "
        "characteristic curves for regional pairwise disagreement (one minus the mean pairwise Dice "
        "of the three seed models), mean predictive entropy, and low predicted regional volume "
        "(minus log[1 + volume in mm3]). Higher scores indicate higher expected error. The diagonal "
        "is the chance reference. Disagreement AUROC is 0.992 for WT and 0.541 for ET-positive "
        "cases. (C, D) Risk-coverage curves retain cases from low to high score; risk is the mean "
        "1-Dice among retained cases. Exact score ties use the expected loss under random ordering "
        "within each tied block. Dashed horizontal lines represent random ranking and dotted curves "
        "an oracle ranking using observed Dice. The WT and ET risk axes have different ranges. "
        "All curves are computed from observed case-level records and reproduce the audited "
        "AUROC and area under the risk-coverage curve. Retained-subset performance is retrospective "
        "and does not represent improved full-cohort segmentation. The ET-positive subset requires "
        "reference labels and therefore is not itself a label-free deployment rule."
    )
    save(fig, name)



if __name__ == '__main__':
    quality()
