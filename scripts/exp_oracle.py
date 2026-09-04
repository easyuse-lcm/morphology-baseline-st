#!/usr/bin/env python3
"""Realisability of post-hoc method selection (Fig. 2c).

Compares the per-section oracle (best method chosen using the held-out
section's own result) against selection rules that use only the other
sections: best mean, best median, best mean rank, and a single global
choice. Reports how much of the oracle gap each rule closes and how often
it identifies the true best method.

Output: oracle_test.csv
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import wilcoxon

R = pd.read_csv("protocol_check.csv")
ARI = R.pivot(index="img_id", columns="model_id", values="a_std")
MORPH = R.groupby("img_id").b_std.first()
sections = list(ARI.index); methods = list(ARI.columns)
print("sections %d, methods %d" % (len(sections), len(methods)))

rng = np.random.default_rng(0)
rows = []
for s in sections:
    others = [o for o in sections if o != s]
    sub = ARI.loc[others]

    m_mean = sub.mean().idxmax()
    m_med = sub.median().idxmax()
    m_rank = sub.rank(axis=1, ascending=False).mean().idxmin()
    m_glob = ARI.mean().idxmax()

    rows.append(dict(
        img_id=s,
        morphology=MORPH[s],
        oracle=ARI.loc[s].max(),
        oracle_method=ARI.loc[s].idxmax(),
        loo_best=ARI.loc[s, m_mean], loo_best_method=m_mean,
        loo_median=ARI.loc[s, m_med],
        loo_rank=ARI.loc[s, m_rank],
        fixed_global=ARI.loc[s, m_glob],
        method_mean=ARI.loc[s].mean(),
        random=float(np.mean([ARI.loc[s, rng.choice(methods)] for _ in range(200)])),
    ))

T = pd.DataFrame(rows).set_index("img_id")
T.to_csv("oracle_test.csv")

print("\n" + "=" * 84)
print("PER-SECTION")
print("=" * 84)
print(T[["morphology", "oracle", "oracle_method", "loo_best", "loo_best_method",
         "loo_rank", "fixed_global", "method_mean"]].round(4).to_string())

print("\n" + "=" * 84)
print("STRATEGY MEANS")
print("=" * 84)
order = ["morphology", "oracle", "loo_best", "loo_median", "loo_rank",
         "fixed_global", "method_mean", "random"]
for c in order:
    print("  %-14s %.4f" % (c, T[c].mean()))

print("\n" + "=" * 84)
print("HOW MUCH OF THE ORACLE GAP DOES EACH ACHIEVABLE RULE CLOSE?")
print("=" * 84)
lo, hi = T.method_mean.mean(), T.oracle.mean()
print("  floor (method mean) = %.4f   ceiling (oracle) = %.4f   gap = %.4f"
      % (lo, hi, hi - lo))
for c in ["loo_best", "loo_median", "loo_rank", "fixed_global"]:
    frac = (T[c].mean() - lo) / (hi - lo) * 100
    print("  %-14s %.4f  -> closes %5.1f%% of the gap" % (c, T[c].mean(), frac))

print("\n" + "=" * 84)
print("MORPHOLOGY vs EACH STRATEGY  (paired, n=8)")
print("=" * 84)
for c in ["oracle", "loo_best", "loo_median", "loo_rank", "fixed_global", "method_mean"]:
    d = T.morphology - T[c]
    print("  vs %-14s diff %+.4f  wins %d/8  p=%.4f"
          % (c, d.mean(), (d > 0).sum(), wilcoxon(T.morphology, T[c]).pvalue))

print("\n" + "=" * 84)
print("WHY THE ORACLE IS NOT REALISABLE")
print("=" * 84)
print("  distinct oracle winners: %d over %d sections" % (T.oracle_method.nunique(), len(T)))
print("  oracle winners:", dict(T.oracle_method))
hit = (T.oracle_method == T.loo_best_method).sum()
print("  LOO rule picked the true winner on %d/%d sections" % (hit, len(T)))
print("  chance rate would be %.1f%%" % (100 / len(methods)))

# does a method's performance on other sections predict its performance here?
cors = []
for s in sections:
    others = [o for o in sections if o != s]
    cors.append(np.corrcoef(ARI.loc[others].mean().values, ARI.loc[s].values)[0, 1])
print("\n  corr(mean ARI on other 7, ARI on held-out) per section:")
print("   ", " ".join("%.2f" % c for c in cors))
print("  mean = %.3f" % np.mean(cors))
print("  -> low correlation means past performance does not predict future")
