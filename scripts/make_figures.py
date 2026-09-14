#!/usr/bin/env python3
"""Publication figures, Nature Portfolio specification.

  width          180 mm double column (89 mm single where noted)
  max height     170 mm
  typography     Helvetica/Arial, 5-7 pt; panel letters 8 pt bold lowercase
  line weights   0.25-1 pt at final size
  colour         colour-blind safe (blue / vermillion / grey / bluish green)
  output         vector PDF (fonts as text, not outlines) + 600 dpi PNG
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, pandas as pd
from pathlib import Path

MM = 1 / 25.4
W2 = 180 * MM          # double column
OUT = Path("figures")
(OUT / "pdf").mkdir(parents=True, exist_ok=True)
(OUT / "png").mkdir(parents=True, exist_ok=True)

TEXT = "#333333"      # all text and axis lines in dark grey, not pure black
PANEL_CASE = "upper"  # "upper" -> A/B/C (Science, Cell, bioRxiv); "lower" -> a/b/c (Nature Portfolio)

plt.rcParams.update({
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 6.5, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "lines.linewidth": 0.75,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # keep fonts editable
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Okabe-Ito colour-blind safe palette
BLUE = "#0072B2"      # morphology
VERM = "#D55E00"      # predicted ST
GREY = "#999999"      # measured ST
GREEN = "#009E73"     # best-per-section
LGREY = "#CCCCCC"

R = pd.read_csv("results/unified_abc.csv")
per = R.groupby("img_id").first()[["ari_morph", "ari_realst"]]
predm = R.groupby("img_id").ari_pred.mean()
SEC = list(per.index)
NICE = lambda i: i.replace("genecoder_i500_j500", "GeneCodeR")


def lab(ax, s, dx=-0.09, dy=1.02):
    s = s.upper() if PANEL_CASE == "upper" else s.lower()
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="right", color=TEXT)


def save(fig, name):
    fig.savefig(OUT / "pdf" / f"{name}.pdf")
    fig.savefig(OUT / "png" / f"{name}.png")
    plt.close(fig)
    print(f"  {name}.pdf + .png")


# ======================================================================
# Figure 1
# ======================================================================
fig = plt.figure(figsize=(W2, 105 * MM))
gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0], hspace=0.55, wspace=0.26,
                      left=0.075, right=0.985, top=0.97, bottom=0.13)

# --- a  schematic -----------------------------------------------------
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 30)

def box(x, y, w, h, txt, ec, fc="white", fs=5.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=TEXT)

def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=5, lw=0.5, color="0.35",
                                 shrinkA=0, shrinkB=1))

box(1, 12.5, 11, 6, "H&E\nimage", TEXT, "#F2F2F2")
box(19, 22, 25, 6, "11 published prediction methods", VERM, "#FCECE3")
box(48, 22, 15, 6, "785 genes", VERM, "#FCECE3")
box(19, 12.5, 25, 6, "pathology foundation model", BLUE, "#E4F0F8")
box(48, 12.5, 15, 6, "768-d embedding", BLUE, "#E4F0F8")
box(19, 3, 25, 6, "measured spatial transcriptomics", GREY, "#F0F1F3")
box(48, 3, 15, 6, "785 genes", GREY, "#F0F1F3")
box(68, 9.5, 14, 12, "identical\nclustering\nprotocol", TEXT)
box(86, 12.5, 12, 6, "ARI", TEXT)

for y in (25, 15.5, 6):
    arr(12, 15.5, 19, y)
    arr(44, y, 48, y)
    arr(63, y, 68, 15.5)
arr(82, 15.5, 86, 15.5)

for y, t, c in ((28.4, "Path A", VERM), (18.9, "Path B", BLUE), (9.4, "Path C", "0.45")):
    ax.text(18.4, y, t, fontsize=5.6, fontweight="bold", color=c, ha="right", va="center")
ax.text(50, 0.4, "reference: pathologist annotation", fontsize=5.5, color="0.4")
ax.text(-0.5, 29.5, "A" if PANEL_CASE == "upper" else "a", fontsize=8,
        fontweight="bold", va="top", color=TEXT)

# --- b  per-section ---------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
x = np.arange(len(SEC)); w = 0.27
ax.bar(x - w, per.ari_morph, w, color=BLUE, lw=0, label="Morphology only")
ax.bar(x, predm.loc[SEC], w, color=VERM, lw=0, label="Predicted ST (mean of 11)")
ax.bar(x + w, per.ari_realst, w, color=GREY, lw=0, label="Measured ST")
ax.set_xticks(x); ax.set_xticklabels(SEC)
ax.set_ylabel("ARI vs pathologist annotation")
ax.set_xlabel("Tissue section")
ax.set_ylim(0, 0.46)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
          handlelength=0.9, handletextpad=0.4, borderpad=0, columnspacing=1.0,
          fontsize=5.4)
lab(ax, "b", dy=1.10)

# --- c  per-method ----------------------------------------------------
bym = R.groupby("model_id").ari_pred.mean().sort_values()
bym.index = [NICE(i) for i in bym.index]
pv = pd.read_csv("results/protocol_check_by_model.csv").set_index("model_id").p_std
pv.index = [NICE(i) for i in pv.index]

ax = fig.add_subplot(gs[1, 1])
y = np.arange(len(bym))
ax.barh(y, bym.values, color=VERM, height=0.62, lw=0)
ax.axvline(per.ari_morph.mean(), color=BLUE, lw=1.0, zorder=3)
ax.annotate("Morphology only\n0.245", xy=(0.252, len(bym) - 1.1),
            fontsize=5.8, color=BLUE, fontweight="bold", va="top")
for i, m in enumerate(bym.index):
    ax.text(bym.values[i] + 0.004, i, "%.3f" % pv.get(m, np.nan),
            va="center", fontsize=5.2, color="0.4")
ax.set_yticks(y); ax.set_yticklabels(bym.index)
ax.set_xlabel("Mean ARI vs pathologist annotation")
ax.set_xlim(0, 0.33)
ax.set_ylim(-0.8, len(bym) - 0.3)
ax.text(0.33, -0.72, "bar labels: paired $P$, n = 8", fontsize=5.2,
        color="0.45", ha="right")
lab(ax, "c", dy=1.10)

save(fig, "Fig1")

# ======================================================================
# Figure 2
# ======================================================================
E = pd.read_csv("results/encoder_ablation.csv")
S = pd.read_csv("results/sensitivity.csv")
O = pd.read_csv("results/oracle_test.csv")

fig = plt.figure(figsize=(W2, 55 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.25, 1.0], wspace=0.42,
                      left=0.065, right=0.99, top=0.90, bottom=0.30)

# --- a  encoders ------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
enc = [("phikon", "Phikon"), ("densenet121", "DenseNet-121"), ("resnet50", "ResNet-50")]
v = [E[c].mean() for c, _ in enc]
e = [E[c].std() / np.sqrt(len(E)) for c, _ in enc]
ax.bar(range(3), v, yerr=e, capsize=1.6, color=BLUE, width=0.55, lw=0,
       error_kw=dict(lw=0.5, capthick=0.5))
ax.axhline(E.pred_mean.mean(), color=VERM, lw=0.8, ls=(0, (3, 2)))
ax.text(2.45, E.pred_mean.mean() - 0.018, "predicted ST", color=VERM,
        fontsize=5.4, ha="right", va="top")
for i in range(3):
    ax.text(i, v[i] + e[i] + 0.010, "%d/8" % int((E[enc[i][0]] > E.pred_mean).sum()),
            ha="center", fontsize=5.4, color="0.35")
ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in enc], rotation=25, ha="right")
ax.set_ylabel("Mean ARI"); ax.set_ylim(0, 0.30)
ax.text(1, -0.093, "pathology FM        ImageNet", fontsize=5.2, color="0.45", ha="center")
lab(ax, "a", dx=-0.20)

# --- b  six configurations -------------------------------------------
ax = fig.add_subplot(gs[0, 1])
cfg = [("all", "km"), ("all", "gmm"), ("all", "leiden"),
       ("no_undet", "km"), ("no_undet", "gmm"), ("no_undet", "leiden")]
mv = [S[S.scope == s]["morph_" + a].mean() for s, a in cfg]
pvv = [S[S.scope == s]["pred_" + a].mean() for s, a in cfg]
x = np.arange(6); w = 0.36
ax.bar(x - w / 2, mv, w, color=BLUE, lw=0, label="Morphology only")
ax.bar(x + w / 2, pvv, w, color=VERM, lw=0, label="Predicted ST")
ax.set_xticks(x)
ax.set_xticklabels(["K-means", "GMM", "Leiden"] * 2, rotation=25, ha="right")
ax.set_ylabel("Mean ARI"); ax.set_ylim(0, 0.44)
ax.axvline(2.5, color="0.8", lw=0.4, ls=":")
for i in range(6):
    ax.text(i, max(mv[i], pvv[i]) + 0.010, "8/8", ha="center", fontsize=5.4, color="0.35")
ax.text(1, -0.135, "all spots", fontsize=5.2, color="0.45", ha="center")
ax.text(4, -0.135, "undetermined removed", fontsize=5.2, color="0.45", ha="center")
ax.legend(frameon=False, loc="upper left", handlelength=0.9, handletextpad=0.5,
          borderpad=0, labelspacing=0.3, fontsize=5.6)
lab(ax, "b", dx=-0.13)

# --- c  selection strategies -----------------------------------------
ax = fig.add_subplot(gs[0, 2])
st = [("morphology", "Morphology\nonly", BLUE), ("oracle", "Oracle", LGREY),
      ("fixed_global", "Fixed\nglobal", VERM), ("loo_best", "Leave-\none-out", VERM),
      ("method_mean", "Method\nmean", VERM)]
ax.bar(range(5), [O[c].mean() for c, _, _ in st], color=[c for _, _, c in st],
       width=0.6, lw=0)
ax.axhline(O.morphology.mean(), color=BLUE, lw=0.5, ls=":")
ax.set_xticks(range(5)); ax.set_xticklabels([n for _, n, _ in st], fontsize=5.4)
ax.set_ylabel("Mean ARI"); ax.set_ylim(0, 0.315)
ax.annotate("not realisable", xy=(1, O.oracle.mean()), xytext=(1.55, 0.283),
            fontsize=5.2, color="0.4", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.55",
                            shrinkA=0, shrinkB=1))
ax.text(3, -0.088, "realisable selection rules", fontsize=5.2, color="0.45", ha="center")
lab(ax, "c", dx=-0.17)

save(fig, "Fig2")

# ======================================================================
# Figure 3
# ======================================================================
N = pd.read_csv("results/nonmorph_reference.csv")
V = pd.read_csv("results/d2_variance.csv")
M = pd.read_csv("results/d1d2_per_method.csv")
M["model_id"] = [NICE(i) for i in M.model_id]

fig = plt.figure(figsize=(W2, 58 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.8], wspace=0.46,
                      left=0.065, right=0.99, top=0.86, bottom=0.20)

# --- a  two references ------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
w = 0.24
series = [("morph_GT1", "morph_GT2", "Morphology only", BLUE),
          ("pred_GT1", "pred_GT2", "Predicted ST (mean)", VERM),
          (None, "predbest_GT2", "Predicted ST (best per section)", GREEN)]
from matplotlib.patches import Patch
for k, (g1, g2, nm, col) in enumerate(series):
    off = (k - 1) * w
    if g1:
        ax.bar([0 + off], [N[g1].mean()], w, color=col, lw=0)
    ax.bar([1 + off], [N[g2].mean()], w, color=col, lw=0)
handles = [Patch(facecolor=c, label=n) for _, _, n, c in series]
ax.set_xticks([0, 1])
ax.set_xticklabels(["Pathologist\nannotation", "Molecular\nreference"])
ax.set_ylabel("Mean ARI"); ax.set_ylim(0, 0.30)
ax.text(0, 0.263, "$P$ = 0.008", ha="center", fontsize=5.6, color="0.35")
ax.text(1, 0.263, "$P$ = 0.15", ha="center", fontsize=5.6, color="0.35")
ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=1,
          handlelength=0.9, handletextpad=0.5, borderpad=0, labelspacing=0.25,
          fontsize=5.4)
ax.text(0.5, -0.085, "derived from image        derived from transcriptome",
        fontsize=5.0, color="0.45", ha="center")
lab(ax, "a", dx=-0.17, dy=1.12)

# --- b  delta ARI -----------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
M = M.sort_values("delta_mean")
y = np.arange(len(M))
ax.errorbar(M.delta_mean, y, xerr=M.delta_sd / np.sqrt(8), fmt="o", ms=2.4,
            color=VERM, mec="none", ecolor="0.6", elinewidth=0.5, capsize=1.0,
            capthick=0.5, ls="none")
ax.axvline(0, color=BLUE, lw=1.0)
ax.set_yticks(y); ax.set_yticklabels(M.model_id)
ax.set_xlabel(r"$\Delta$ARI (method $-$ morphology)")
ax.set_xlim(-0.21, 0.045)
ax.text(0.006, len(M) - 0.6, "morphology\nbaseline", color=BLUE, fontsize=5.6, va="top")
lab(ax, "b", dx=-0.22, dy=1.12)

# --- c  variance decomposition ---------------------------------------
ax = fig.add_subplot(gs[0, 2])
key = {"raw ARI": "Raw ARI", "ARI - morphology (delta)": r"$\Delta$ARI",
       "ARI - measured ST (control)": "Control"}
V2 = V[V.metric.isin(key)].copy()
V2["nm"] = [key[m] for m in V2.metric]
V2 = V2.set_index("nm").loc[["Raw ARI", r"$\Delta$ARI", "Control"]]
ax.bar(range(3), V2.icc * 100, color=["0.55", BLUE, LGREY], width=0.58, lw=0)
ax.set_xticks(range(3)); ax.set_xticklabels(V2.index)
ax.set_ylabel("Variance explained\nby tissue section (%)")
ax.set_ylim(0, 74)
for i, v in enumerate(V2.icc * 100):
    ax.text(i, v + 1.8, "%.0f" % v, ha="center", fontsize=6)
ax.text(2, -13, "$-$ measured ST", fontsize=5.0, color="0.45", ha="center")
lab(ax, "c", dx=-0.30, dy=1.12)

save(fig, "Fig3")
print("\nAll figures in ./figures/ — 180 mm wide, vector PDF + 600 dpi PNG")
