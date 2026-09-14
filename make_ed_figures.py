#!/usr/bin/env python3
"""Extended Data figures 1-4 (same specification as main figures)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path

MM = 1 / 25.4
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
TEXT = "#333333"
PANEL_CASE = "upper"

plt.rcParams.update({
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 5.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "lines.linewidth": 0.75,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE, VERM, GREY, GREEN, LGREY = "#0072B2", "#D55E00", "#999999", "#009E73", "#CCCCCC"
NICE = lambda i: i.replace("genecoder_i500_j500", "GeneCodeR")


def lab(ax, s, dx=-0.16, dy=1.02):
    s = s.upper() if PANEL_CASE == "upper" else s.lower()
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right", color=TEXT)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig)
    print(f"  {name}")


# ======================================================================
# ED Fig 1 - protocol robustness: standardised vs non-standardised
# ======================================================================
P = pd.read_csv("results/protocol_check.csv")
pp = P.groupby("img_id").first()
fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 52 * MM))
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.22, wspace=0.42)
for ax, (xs, ys, nm, col, n) in zip(axes, [
        ("b_std", "b_nostd", "Morphology only", BLUE, pp),
        ("c_std", "c_nostd", "Measured ST", GREY, pp),
        ("a_std", "a_nostd", "Predicted ST (88 method–section)", VERM, P)]):
    ax.scatter(n[xs], n[ys], s=9, color=col, alpha=0.85, lw=0)
    lim = [0, max(n[xs].max(), n[ys].max()) * 1.08]
    ax.plot(lim, lim, color="0.7", lw=0.5, ls=":")
    r = np.corrcoef(n[xs], n[ys])[0, 1]
    ax.text(0.04, 0.93, "r = %.2f" % r, transform=ax.transAxes, fontsize=6)
    ax.set_xlabel("ARI, standardised"); ax.set_ylabel("ARI, centred only")
    ax.set_title(nm, fontsize=6.5, loc="left")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
for ax, s in zip(axes, "abc"):
    lab(ax, s, dx=-0.28)
save(fig, "ED_Fig1")

# ======================================================================
# ED Fig 2 - ridge regularisation sensitivity
# ======================================================================
A = pd.read_csv("results/alpha_grid.csv")
Ai = pd.read_csv("results/alpha_innercv.csv")
g = A.groupby("alpha").agg(m=("ari_bottleneck", "mean"), s=("ari_bottleneck", "std"),
                           morph=("ari_morph", "mean"))
fig, ax = plt.subplots(figsize=(89 * MM, 55 * MM))
fig.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.20)
ax.errorbar(g.index, g.m, yerr=g.s / np.sqrt(8), fmt="o-", ms=2.6, color=VERM,
            mec="none", ecolor="0.6", elinewidth=0.5, capsize=1.2, capthick=0.5, lw=0.8,
            label="ridge to 785 genes, then cluster")
ax.axhline(g.morph.iloc[0], color=BLUE, lw=1.0, label="morphology only")
ax.axvline(Ai.alpha_sel.mode().iloc[0], color="0.5", lw=0.5, ls=(0, (3, 2)))
ax.text(Ai.alpha_sel.mode().iloc[0] * 1.3, 0.262, "selected by\ninner CV", fontsize=5.4, color="0.45")
ax.set_xscale("log"); ax.set_xlabel(r"ridge penalty $\alpha$"); ax.set_ylabel("Mean ARI")
ax.set_ylim(0.14, 0.27)
ax.legend(frameon=False, loc="lower left", handlelength=1.2)
save(fig, "ED_Fig2")

# ======================================================================
# ED Fig 3 - rank-matched and target-shuffled controls
# ======================================================================
RM = pd.read_csv("results/rank_matched_control.csv")
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 55 * MM), gridspec_kw={"width_ratios": [0.55, 1]})
fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.30, wspace=0.35)
ax = axes[0]
ax.bar([0, 1], [RM.rank_morph.mean(), RM.rank_ridge.mean()], color=[BLUE, VERM], width=0.55, lw=0)
for i, v in enumerate([RM.rank_morph.mean(), RM.rank_ridge.mean()]):
    ax.text(i, v + 3, "%.0f" % v, ha="center", fontsize=6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Morphology\nembedding", "Ridge\noutput"])
ax.set_ylabel("Effective rank\n(PCs for 95% variance)"); ax.set_ylim(0, 125)
lab(ax, "a", dx=-0.36)
ax = axes[1]
cond = [("morph", "Morphology\nonly", BLUE), ("morph_rank", "Morphology\ntruncated\nto rank 24", BLUE),
        ("ridge_shuf", "Ridge onto\npermuted\ntargets", LGREY), ("ridge_rand", "Ridge onto\nrandom\ntargets", LGREY),
        ("ridge", "Ridge onto\nmeasured\nexpression", VERM)]
v = [RM[c].mean() for c, _, _ in cond]; e = [RM[c].std() / np.sqrt(8) for c, _, _ in cond]
ax.bar(range(5), v, yerr=e, color=[c for _, _, c in cond], width=0.6, lw=0, capsize=1.6,
       error_kw=dict(lw=0.5, capthick=0.5))
ax.set_xticks(range(5)); ax.set_xticklabels([n for _, n, _ in cond], fontsize=5.4)
ax.set_ylabel("Mean ARI"); ax.set_ylim(0, 0.30)
ax.axhline(v[0], color=BLUE, lw=0.5, ls=":")
lab(ax, "b", dx=-0.12)
save(fig, "ED_Fig3")

# ======================================================================
# ED Fig 4 - permutation null per section
# ======================================================================
PN = pd.read_csv("results/permutation_null.csv")
fig, ax = plt.subplots(figsize=(89 * MM, 58 * MM))
fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.20)
x = np.arange(len(PN))
ax.errorbar(x - 0.12, PN.nullA_mean, yerr=2 * PN.nullA_sd, fmt="s", ms=3, color=LGREY,
            mec="0.5", mew=0.4, ecolor="0.6", elinewidth=0.6, capsize=1.6, capthick=0.5,
            ls="none", label="permuted-target null (mean ± 2 s.d., 100 draws)")
ax.scatter(x + 0.12, PN.ridge_real, s=14, color=VERM, zorder=3, lw=0, label="ridge onto measured expression")
ax.scatter(x, PN.morph, s=14, color=BLUE, marker="_", lw=1.2, zorder=3, label="morphology only")
for i, p in enumerate(PN.p_permA):
    ax.text(i, -0.03, "%.2f" % p if p >= 0.01 else "<0.01", ha="center", fontsize=5.2,
            color=VERM if p < 0.05 else "0.5")
ax.text(len(PN) - 0.5, -0.03, "P", ha="left", fontsize=5.2, color="0.45", style="italic")
ax.set_xticks(x); ax.set_xticklabels(PN.img_id); ax.set_xlabel("Tissue section")
ax.set_ylabel("ARI"); ax.set_ylim(-0.05, 0.45)
ax.axhline(0, color="0.85", lw=0.4)
ax.legend(frameon=False, loc="upper left", handlelength=1.2, fontsize=5.4)
save(fig, "ED_Fig4")
print("\nExtended Data figures written to ./figures/")
