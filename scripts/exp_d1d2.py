#!/usr/bin/env python3
"""Baseline normalisation and variance decomposition (Fig. 3b,c).

Computes the share of variance in ARI attributable to section identity
before and after subtracting the per-section morphology baseline, with
subtraction of the measured-expression ARI as a control. Also reports
ranking agreement across protocols for raw and normalised metrics.

Output: d1_ranking_stability.csv, d2_variance.csv, d1d2_per_method.csv
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr

R = pd.read_csv("protocol_check.csv")
print("loaded protocol_check.csv:", R.shape)
print("methods:", R.model_id.nunique(), " sections:", R.img_id.nunique())

EPS = 1e-6
for suf in ["std", "nostd"]:
    R["delta_" + suf] = R["a_" + suf] - R["b_" + suf]
    R["ratio_" + suf] = R["a_" + suf] / (R["b_" + suf] + EPS)
    R["dreal_" + suf] = R["a_" + suf] - R["c_" + suf]

print("\n" + "=" * 78)
print("D1  RANKING STABILITY ACROSS PROTOCOLS")
print("=" * 78)
print("%-28s %10s %10s" % ("metric", "Spearman", "Kendall"))
from scipy.stats import kendalltau
d1 = []
for name, key in [("raw ARI", "a"), ("ARI - morphology (delta)", "delta"),
                  ("ARI / morphology (ratio)", "ratio"),
                  ("ARI - measured ST (control)", "dreal")]:
    s = R.groupby("model_id")[key + "_std"].mean()
    t = R.groupby("model_id")[key + "_nostd"].mean()
    t = t.loc[s.index]
    rho = spearmanr(s, t).statistic
    tau = kendalltau(s, t).statistic
    d1.append(dict(metric=name, spearman=rho, kendall=tau))
    print("%-28s %10.3f %10.3f" % (name, rho, tau))

print("\n  -> if delta/ratio > raw, baseline normalisation stabilises ranking")

print("\n" + "=" * 78)
print("D2  BETWEEN-SECTION VARIANCE OF METHOD PERFORMANCE")
print("=" * 78)
print("%-28s %12s %12s %12s" % ("metric", "mean SD", "mean CV", "ICC(section)"))


def icc_section(df, col):
    """fraction of total variance attributable to section (one-way ANOVA)."""
    g = df.groupby("img_id")[col]
    k = df.model_id.nunique()
    grand = df[col].mean()
    ss_between = (g.mean() - grand).pow(2).mul(g.size()).sum()
    ss_total = (df[col] - grand).pow(2).sum()
    return ss_between / ss_total


d2 = []
for name, key in [("raw ARI", "a"), ("ARI - morphology (delta)", "delta"),
                  ("ARI / morphology (ratio)", "ratio"),
                  ("ARI - measured ST (control)", "dreal")]:
    col = key + "_std"
    sd = R.groupby("model_id")[col].std().mean()
    cv = (R.groupby("model_id")[col].std() / R.groupby("model_id")[col].mean().abs()).mean()
    icc = icc_section(R, col)
    d2.append(dict(metric=name, sd=sd, cv=cv, icc=icc))
    print("%-28s %12.4f %12.4f %12.3f" % (name, sd, cv, icc))

print("\n  mean SD  = average across methods of the SD over the 8 sections")
print("  ICC      = share of total variance explained by WHICH SECTION it is")
print("  -> lower ICC means section difficulty has been absorbed, so more of")
print("     the remaining variance reflects the method itself")

print("\n" + "=" * 78)
print("D2b  HOW MUCH DOES SECTION IDENTITY DOMINATE RAW ARI?")
print("=" * 78)
print("  raw ARI:   section explains %.1f%% of variance" % (100 * d2[0]["icc"]))
print("  delta:     section explains %.1f%% of variance" % (100 * d2[1]["icc"]))
red = (d2[0]["icc"] - d2[1]["icc"]) / d2[0]["icc"] * 100
print("  -> reduction: %.1f%%" % red)

print("\n" + "=" * 78)
print("PER-METHOD TABLE (standardised protocol)")
print("=" * 78)
tab = R.groupby("model_id").agg(
    raw_mean=("a_std", "mean"), raw_sd=("a_std", "std"),
    delta_mean=("delta_std", "mean"), delta_sd=("delta_std", "std"),
).sort_values("raw_mean", ascending=False)
tab["rank_raw_std"] = tab.raw_mean.rank(ascending=False).astype(int)
nost = R.groupby("model_id").a_nostd.mean()
tab["rank_raw_nostd"] = nost.loc[tab.index].rank(ascending=False).astype(int)
dn = R.groupby("model_id").delta_nostd.mean()
tab["rank_delta_std"] = tab.delta_mean.rank(ascending=False).astype(int)
tab["rank_delta_nostd"] = dn.loc[tab.index].rank(ascending=False).astype(int)
tab["rank_shift_raw"] = (tab.rank_raw_std - tab.rank_raw_nostd).abs()
tab["rank_shift_delta"] = (tab.rank_delta_std - tab.rank_delta_nostd).abs()
print(tab.round(4).to_string())
print("\n  mean absolute rank shift between protocols:")
print("    raw ARI  = %.2f positions" % tab.rank_shift_raw.mean())
print("    delta    = %.2f positions" % tab.rank_shift_delta.mean())

pd.DataFrame(d1).to_csv("d1_ranking_stability.csv", index=False)
pd.DataFrame(d2).to_csv("d2_variance.csv", index=False)
tab.to_csv("d1d2_per_method.csv")
print("\nsaved d1_ranking_stability.csv / d2_variance.csv / d1d2_per_method.csv")
