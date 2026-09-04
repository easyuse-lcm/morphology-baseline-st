#!/usr/bin/env python3
"""Robustness to the downstream protocol (Fig. 2b).

Recomputes the three-path comparison with and without the per-feature
standardisation step, and reports per-method paired tests and rank
agreement between the two protocols.

Output: protocol_check.csv, protocol_check_by_model.csv
"""
import gzip, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.stats import wilcoxon
import pyreadr

ANNOT = ["A1","B1","C1","D1","E1","F1","G2","H1"]
ROOT = Path("her2st")


def ari(X, y, K, std=True, n_pcs=50, seeds=range(20)):
    X = np.asarray(X, float)
    X = X[:, np.nanstd(X, 0) > 1e-12]
    X = np.nan_to_num(X)
    Xp = StandardScaler().fit_transform(X) if std else (X - X.mean(0))
    Z = PCA(min(n_pcs, Xp.shape[1], Xp.shape[0]-1), random_state=0).fit_transform(Xp)
    return float(np.mean([adjusted_rand_score(y, KMeans(K, n_init=10, random_state=s).fit_predict(Z))
                          for s in seeds]))


df = pyreadr.read_r("her2st_cluster_11.rds")[None]
meta = ["model_id","img_id","patch_id","label","cluster","gt_cluster","cluster_observed"]
genes = [c for c in df.columns if c not in meta]
cache = dict(np.load("morph_cache.npz", allow_pickle=True))

rows = []
for img in ANNOT:
    sub = df[df.img_id == img]
    ref = sub[sub.model_id == sub.model_id.iloc[0]].drop_duplicates("patch_id").reset_index(drop=True)
    y_all = ref.gt_cluster.astype(str).values
    sid = ref.patch_id.str.split("_").str[1].values
    K = len(np.unique(y_all))

    # real ST on the same 785 genes
    with gzip.open(next(ROOT.glob("data/ST-cnts/%s.tsv.gz" % img)), "rt") as f:
        cnt = pd.read_csv(f, sep="\t", index_col=0)
    keep = [g for g in genes if g in cnt.columns]
    X = cnt[keep].to_numpy(float); lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
    E = pd.DataFrame(np.log1p(X / lib * 1e4), index=cnt.index, columns=keep)
    okC = np.array([s in E.index for s in sid])
    Ec = E.loc[[s for s, o in zip(sid, okC) if o]].to_numpy()
    c_std = ari(Ec, y_all[okC], K, True)
    c_nostd = ari(Ec, y_all[okC], K, False)

    # morphology
    M = cache[img]; okB = ~np.isnan(M).any(1)
    b_std = ari(M[okB], y_all[okB], K, True)
    b_nostd = ari(M[okB], y_all[okB], K, False)

    pos = {p: i for i, p in enumerate(ref.patch_id)}
    for mod, g in sub.groupby("model_id"):
        g = g.drop_duplicates("patch_id").set_index("patch_id")
        kp = [p for p in ref.patch_id if p in g.index]
        ii = np.array([pos[p] for p in kp])
        P = g.loc[kp, genes].to_numpy()
        rows.append(dict(img_id=img, model_id=mod, K=K, n=len(kp),
                         a_std=ari(P, y_all[ii], K, True),
                         a_nostd=ari(P, y_all[ii], K, False),
                         b_std=b_std, b_nostd=b_nostd,
                         c_std=c_std, c_nostd=c_nostd))
    print("[done] %s  B %.4f/%.4f   C %.4f/%.4f  (std/nostd)"
          % (img, b_std, b_nostd, c_std, c_nostd), flush=True)

R = pd.DataFrame(rows); R.to_csv("protocol_check.csv", index=False)

print("\n" + "=" * 82)
print("R1  morphology vs each published method")
print("=" * 82)
print("%-22s %8s %8s | %8s %8s | %s" % ("method", "A_std", "A_nostd", "p_std", "p_nostd", "wins std/nostd"))
out = []
for mod, g in R.groupby("model_id"):
    g = g.sort_values("img_id")
    p1 = wilcoxon(g.b_std, g.a_std).pvalue
    p2 = wilcoxon(g.b_nostd, g.a_nostd).pvalue
    w1 = (g.b_std > g.a_std).sum(); w2 = (g.b_nostd > g.a_nostd).sum()
    out.append(dict(model_id=mod, a_std=g.a_std.mean(), a_nostd=g.a_nostd.mean(),
                    p_std=p1, p_nostd=p2, w_std=w1, w_nostd=w2))
    print("%-22s %8.4f %8.4f | %8.4f %8.4f | %d/8  %d/8"
          % (mod, g.a_std.mean(), g.a_nostd.mean(), p1, p2, w1, w2))
O = pd.DataFrame(out); O.to_csv("protocol_check_by_model.csv", index=False)

print("\n  morphology mean:  std %.4f   nostd %.4f" % (R.b_std.mean(), R.b_nostd.mean()))
print("  predicted  mean:  std %.4f   nostd %.4f" % (R.a_std.mean(), R.a_nostd.mean()))
print("  real ST    mean:  std %.4f   nostd %.4f" % (R.c_std.mean(), R.c_nostd.mean()))
print("\n  methods beaten by morphology:  std %d/11   nostd %d/11"
      % ((O.w_std >= 5).sum(), (O.w_nostd >= 5).sum()))
print("  methods with p<0.05:           std %d/11   nostd %d/11"
      % ((O.p_std < 0.05).sum(), (O.p_nostd < 0.05).sum()))

print("\n" + "=" * 82)
print("R2  morphology vs real ST  (per slide, n=8)")
print("=" * 82)
per = R.groupby("img_id").first()
for tag, b, c in [("std", "b_std", "c_std"), ("nostd", "b_nostd", "c_nostd")]:
    d = per[b] - per[c]
    print("  %-6s morph %.4f  realST %.4f  diff %+.4f  wins %d/8  p=%.4f"
          % (tag, per[b].mean(), per[c].mean(), d.mean(), (d > 0).sum(),
             wilcoxon(per[b], per[c]).pvalue))

print("\n" + "=" * 82)
print("R3  predicted ST vs real ST  (the original papers' claim)")
print("=" * 82)
for tag, a, c in [("std", "a_std", "c_std"), ("nostd", "a_nostd", "c_nostd")]:
    d = R[a] - R[c]
    print("  %-6s predicted %.4f  realST %.4f  diff %+.4f  wins %d/88 (%.1f%%)  p=%.4f"
          % (tag, R[a].mean(), R[c].mean(), d.mean(), (d > 0).sum(),
             (d > 0).mean()*100, wilcoxon(R[a], R[c]).pvalue))
print("  (note: 88-pair p is pseudo-replicated; reported for comparison only)")

print("\n" + "=" * 82)
print("AGREEMENT BETWEEN PROTOCOLS")
print("=" * 82)
print("  corr(A_std, A_nostd)  over 88 pairs = %.3f" % np.corrcoef(R.a_std, R.a_nostd)[0,1])
print("  corr(B_std, B_nostd)  over 8 slides = %.3f" % np.corrcoef(per.b_std, per.b_nostd)[0,1])
print("  corr(C_std, C_nostd)  over 8 slides = %.3f" % np.corrcoef(per.c_std, per.c_nostd)[0,1])
print("\n  method ranking agreement (Spearman) = %.3f"
      % O[["a_std","a_nostd"]].corr(method="spearman").iloc[0,1])
