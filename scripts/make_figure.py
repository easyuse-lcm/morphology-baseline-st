#!/usr/bin/env python3
"""Main figure: per-section three-path ARI and per-method comparison.

Input:  unified_abc.csv
Output: figure_main.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

R = pd.read_csv("unified_abc.csv")
per = R.groupby("img_id").first()[["ari_morph", "ari_realst"]]
pred = R.groupby("img_id").ari_pred.mean()
bym = R.groupby("model_id").ari_pred.mean().sort_values()
bym.index = [i.replace("genecoder_i500_j500", "GeneCodeR") for i in bym.index]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ---- Panel A: per slide, three paths ----
slides = list(per.index)
x = np.arange(len(slides)); w = 0.27
ax1.bar(x - w, per.ari_morph, w, label="Morphology only (path B)", color="#2c5f8a")
ax1.bar(x,      pred.loc[slides], w, label="Predicted ST (path A, mean of 11)", color="#c96a3f")
ax1.bar(x + w, per.ari_realst, w, label="Measured ST (path C)", color="#9aa6b2")
ax1.set_xticks(x); ax1.set_xticklabels(slides)
ax1.set_ylabel("ARI vs pathologist annotation")
ax1.set_xlabel("HER2+ breast cancer section")
ax1.set_title("A   Morphology alone beats measured ST on all 8 sections (p = 0.008)", loc="left", fontsize=10)
ax1.legend(frameon=False, fontsize=8, loc="upper left")
ax1.set_ylim(0, 0.45)

# ---- Panel B: 11 methods vs morphology ----
y = np.arange(len(bym))
ax2.barh(y, bym.values, color="#c96a3f", height=0.62)
ax2.axvline(per.ari_morph.mean(), color="#2c5f8a", lw=2)
ax2.text(per.ari_morph.mean() + 0.006, len(bym) - 0.4,
         "Morphology only\n0.245", color="#2c5f8a", fontsize=8.5, va="top", fontweight="bold")
ax2.set_yticks(y); ax2.set_yticklabels(bym.index, fontsize=8.5)
ax2.set_xlabel("ARI vs pathologist annotation")
ax2.set_title("B   All 11 published methods fall below the baseline", loc="left", fontsize=10)
ax2.set_xlim(0, 0.30)

fig.tight_layout()
fig.savefig("figure_main.png", bbox_inches="tight")
print("saved figure_main.png")
print("panel A slides:", slides)
print("morphology mean %.4f | predicted mean %.4f | measured mean %.4f"
      % (per.ari_morph.mean(), R.ari_pred.mean(), per.ari_realst.mean()))
