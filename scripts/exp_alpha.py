#!/usr/bin/env python3
"""Regularisation sensitivity for the ridge bottleneck analysis.

Fits a ridge regression from morphology embeddings to the 785 genes under
leave-one-section-out cross-validation across a grid of penalties, and
under an inner cross-validated choice of penalty.

Output: alpha_grid.csv, alpha_innercv.csv
"""
import gzip, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, r2_score
from scipy.stats import wilcoxon
import pyreadr

ANNOT = ["A1","B1","C1","D1","E1","F1","G2","H1"]
ROOT = Path("her2st")
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]


def cl_ari(X, y, K, n_pcs=50, seeds=range(20)):
    X = np.asarray(X, float)
    X = X[:, np.nanstd(X, 0) > 1e-12]
    X = np.nan_to_num(X)
    Z = PCA(min(n_pcs, X.shape[1], X.shape[0]-1), random_state=0).fit_transform(
        StandardScaler().fit_transform(X))
    return float(np.mean([adjusted_rand_score(y, KMeans(K, n_init=10, random_state=s).fit_predict(Z))
                          for s in seeds]))


df = pyreadr.read_r("her2st_cluster_11.rds")[None]
meta = ["model_id","img_id","patch_id","label","cluster","gt_cluster","cluster_observed"]
genes = [c for c in df.columns if c not in meta]
cache = dict(np.load("morph_cache.npz", allow_pickle=True))

D = {}
for img in ANNOT:
    ref = df[df.img_id == img]
    ref = ref[ref.model_id == ref.model_id.iloc[0]].drop_duplicates("patch_id").reset_index(drop=True)
    y = ref.gt_cluster.astype(str).values
    sid = ref.patch_id.str.split("_").str[1].values
    with gzip.open(next(ROOT.glob("data/ST-cnts/%s.tsv.gz" % img)), "rt") as f:
        cnt = pd.read_csv(f, sep="\t", index_col=0)
    keep = [g for g in genes if g in cnt.columns]
    X = cnt[keep].to_numpy(float)
    lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
    E = pd.DataFrame(np.log1p(X / lib * 1e4), index=cnt.index, columns=keep)
    M = cache[img]
    ok = (~np.isnan(M).any(1)) & np.array([s in E.index for s in sid])
    D[img] = dict(M=M[ok],
                  E=E.loc[[s for s, o in zip(sid, ok) if o]].to_numpy(),
                  y=y[ok], K=len(np.unique(y)))

# ---------- 1. fixed alpha grid ----------
rows = []
for img in ANNOT:
    tr = [i for i in ANNOT if i != img]
    Xtr = np.vstack([D[i]["M"] for i in tr]); Ytr = np.vstack([D[i]["E"] for i in tr])
    sx = StandardScaler().fit(Xtr); Xt = sx.transform(Xtr); Xte = sx.transform(D[img]["M"])
    am = cl_ari(D[img]["M"], D[img]["y"], D[img]["K"])
    for a in ALPHAS:
        P = Ridge(alpha=a).fit(Xt, Ytr).predict(Xte)
        rows.append(dict(img_id=img, alpha=a, ari_morph=am,
                         ari_bottleneck=cl_ari(P, D[img]["y"], D[img]["K"]),
                         r2=r2_score(D[img]["E"], P, multioutput="variance_weighted"),
                         pred_sd=float(P.std())))
    print("[grid] %s done" % img, flush=True)

G = pd.DataFrame(rows); G.to_csv("alpha_grid.csv", index=False)

print("\n" + "=" * 78)
print("ALPHA SENSITIVITY: bottleneck ARI per alpha")
print("=" * 78)
piv = G.pivot(index="img_id", columns="alpha", values="ari_bottleneck")
piv["morph_baseline"] = G.groupby("img_id").ari_morph.first()
print(piv.round(4).to_string())

print("\nSummary:")
s = G.groupby("alpha").agg(ari_bottleneck=("ari_bottleneck", "mean"), r2=("r2", "mean"))
s["morph"] = G.ari_morph.mean()
s["loss_pct"] = (s["morph"] - s.ari_bottleneck) / s["morph"] * 100
for a in ALPHAS:
    g = G[G.alpha == a]
    s.loc[a, "wins"] = "%d/8" % (g.ari_morph > g.ari_bottleneck).sum()
    s.loc[a, "p"] = wilcoxon(g.ari_morph, g.ari_bottleneck).pvalue
print(s.round(4).to_string())

# ---------- 2. inner-CV alpha selection ----------
print("\n" + "=" * 78)
print("INNER-CV ALPHA SELECTION (fairest to the bottleneck)")
print("=" * 78)
rows2 = []
for img in ANNOT:
    tr = [i for i in ANNOT if i != img]
    best = (None, -9e9)
    for a in ALPHAS:
        sc = []
        for v in tr:
            it = [i for i in tr if i != v]
            Xi = np.vstack([D[i]["M"] for i in it]); Yi = np.vstack([D[i]["E"] for i in it])
            sx = StandardScaler().fit(Xi)
            P = Ridge(alpha=a).fit(sx.transform(Xi), Yi).predict(sx.transform(D[v]["M"]))
            sc.append(r2_score(D[v]["E"], P, multioutput="variance_weighted"))
        if np.mean(sc) > best[1]:
            best = (a, np.mean(sc))
    a = best[0]
    Xtr = np.vstack([D[i]["M"] for i in tr]); Ytr = np.vstack([D[i]["E"] for i in tr])
    sx = StandardScaler().fit(Xtr)
    P = Ridge(alpha=a).fit(sx.transform(Xtr), Ytr).predict(sx.transform(D[img]["M"]))
    rows2.append(dict(img_id=img, alpha_sel=a,
                      ari_morph=cl_ari(D[img]["M"], D[img]["y"], D[img]["K"]),
                      ari_bottleneck=cl_ari(P, D[img]["y"], D[img]["K"])))
    print("  %s: alpha=%g  morph=%.4f  bottleneck=%.4f" %
          (img, a, rows2[-1]["ari_morph"], rows2[-1]["ari_bottleneck"]), flush=True)

B = pd.DataFrame(rows2); B.to_csv("alpha_innercv.csv", index=False)
d = B.ari_morph - B.ari_bottleneck
print("\n  morph %.4f  vs  bottleneck %.4f" % (B.ari_morph.mean(), B.ari_bottleneck.mean()))
print("  loss %+.4f (%.1f%%)  wins %d/8  p=%.4f" %
      (d.mean(), d.mean() / B.ari_morph.mean() * 100, (d > 0).sum(),
       wilcoxon(B.ari_morph, B.ari_bottleneck).pvalue))

# ---------- 3. oracle alpha per slide ----------
best_per = G.loc[G.groupby("img_id").ari_bottleneck.idxmax()]
d2 = best_per.ari_morph - best_per.ari_bottleneck
print("\n" + "=" * 78)
print("ORACLE: best alpha picked post-hoc per slide (not achievable in practice)")
print("=" * 78)
print(best_per[["img_id","alpha","ari_morph","ari_bottleneck"]].round(4).to_string(index=False))
print("\n  morph %.4f vs bottleneck-oracle %.4f" % (best_per.ari_morph.mean(), best_per.ari_bottleneck.mean()))
print("  loss %+.4f  wins %d/8  p=%.4f" %
      (d2.mean(), (d2 > 0).sum(), wilcoxon(best_per.ari_morph, best_per.ari_bottleneck).pvalue))
