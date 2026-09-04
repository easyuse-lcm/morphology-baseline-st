#!/usr/bin/env python3
"""Dimension- and rank-matched controls for the bottleneck analysis.

Compares the ridge output against: morphology truncated to the same
effective rank; ridge fitted to permuted targets; and ridge fitted to
Gaussian random targets.

Output: rank_matched_control.csv
"""
import gzip, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from scipy.stats import wilcoxon
import pyreadr

ANNOT = ["A1","B1","C1","D1","E1","F1","G2","H1"]
ROOT = Path("her2st")
ALPHA = 1e4


def ari(X, y, K, std=True, n_pcs=50, seeds=range(20)):
    X = np.asarray(X, float)
    X = X[:, np.nanstd(X, 0) > 1e-12]
    X = np.nan_to_num(X)
    Xp = StandardScaler().fit_transform(X) if std else (X - X.mean(0))
    Z = PCA(min(n_pcs, Xp.shape[1], Xp.shape[0]-1), random_state=0).fit_transform(Xp)
    return float(np.mean([adjusted_rand_score(y, KMeans(K, n_init=10, random_state=s).fit_predict(Z))
                          for s in seeds]))


def eff_rank(X, thresh=0.95):
    """number of PCs needed to explain `thresh` of variance, and participation ratio"""
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    v = s**2
    v = v / v.sum()
    n95 = int(np.searchsorted(np.cumsum(v), thresh) + 1)
    pr = float((v.sum()**2) / (v**2).sum())      # participation ratio
    return n95, pr


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
    # fix R name mangling: HLA.DRA -> HLA-DRA
    colmap = {c: c for c in cnt.columns}
    for c in cnt.columns:
        if "-" in c:
            colmap[c] = c.replace("-", ".")
    cnt = cnt.rename(columns=colmap)
    keep = [g for g in genes if g in cnt.columns]
    X = cnt[keep].to_numpy(float); lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
    E = pd.DataFrame(np.log1p(X / lib * 1e4), index=cnt.index, columns=keep)
    M = cache[img]
    ok = (~np.isnan(M).any(1)) & np.array([s in E.index for s in sid])
    D[img] = dict(M=M[ok], E=E.loc[[s for s, o in zip(sid, ok) if o]].to_numpy(),
                  y=y[ok], K=len(np.unique(y)), ngene=len(keep))

print("gene match after HLA fix: %d / %d" % (D["A1"]["ngene"], len(genes)))
print()

rows = []
for img in ANNOT:
    tr = [i for i in ANNOT if i != img]
    Xtr = np.vstack([D[i]["M"] for i in tr]); Ytr = np.vstack([D[i]["E"] for i in tr])
    sx = StandardScaler().fit(Xtr)
    Xt = sx.transform(Xtr); Xte = sx.transform(D[img]["M"])
    y, K = D[img]["y"], D[img]["K"]
    rng = np.random.default_rng(0)

    P_ridge = Ridge(alpha=ALPHA).fit(Xt, Ytr).predict(Xte)
    P_shuf = Ridge(alpha=ALPHA).fit(Xt, Ytr[rng.permutation(len(Ytr))]).predict(Xte)
    P_rand = Ridge(alpha=ALPHA).fit(Xt, rng.standard_normal(Ytr.shape)).predict(Xte)

    r95_m, pr_m = eff_rank(D[img]["M"])
    r95_r, pr_r = eff_rank(P_ridge)

    # rank-matched morphology: truncate raw features to r95_r PCs
    pm = PCA(n_components=min(r95_r, Xte.shape[1], Xte.shape[0]-1), random_state=0).fit(Xt)
    M_rank = pm.transform(Xte)

    rows.append(dict(img_id=img, K=K, n=len(y),
                     rank_morph=r95_m, rank_ridge=r95_r, pr_morph=pr_m, pr_ridge=pr_r,
                     morph=ari(D[img]["M"], y, K),
                     morph_rank=ari(M_rank, y, K),
                     ridge=ari(P_ridge, y, K),
                     ridge_shuf=ari(P_shuf, y, K),
                     ridge_rand=ari(P_rand, y, K)))
    print("  %-4s rank95 morph=%3d ridge=%3d | morph=%.4f morph_rank=%.4f ridge=%.4f shuf=%.4f rand=%.4f"
          % (img, r95_m, r95_r, rows[-1]["morph"], rows[-1]["morph_rank"],
             rows[-1]["ridge"], rows[-1]["ridge_shuf"], rows[-1]["ridge_rand"]), flush=True)

R = pd.DataFrame(rows); R.to_csv("rank_matched_control.csv", index=False)

print("\n" + "=" * 76)
print("EFFECTIVE RANK (PCs for 95% variance)")
print("=" * 76)
print("  morphology mean = %.1f      ridge output mean = %.1f" % (R.rank_morph.mean(), R.rank_ridge.mean()))
print("  participation ratio: morph %.1f  ridge %.1f" % (R.pr_morph.mean(), R.pr_ridge.mean()))

print("\n" + "=" * 76)
print("ARI BY CONDITION")
print("=" * 76)
for c, name in [("morph","raw morphology"), ("morph_rank","morphology truncated to ridge rank"),
                ("ridge","ridge -> real genes"), ("ridge_shuf","ridge -> shuffled genes"),
                ("ridge_rand","ridge -> random targets")]:
    print("  %-38s %.4f" % (name, R[c].mean()))

print("\n" + "=" * 76)
print("DECOMPOSITION OF THE LOSS")
print("=" * 76)
tot = R.morph - R.ridge
rank_part = R.morph - R.morph_rank
gene_part = R.ridge_shuf - R.ridge
print("  total loss (morph - ridge)            = %+.4f  wins %d/8  p=%.4f"
      % (tot.mean(), (tot > 0).sum(), wilcoxon(R.morph, R.ridge).pvalue))
print("  attributable to RANK COLLAPSE         = %+.4f  wins %d/8  p=%.4f"
      % (rank_part.mean(), (rank_part > 0).sum(), wilcoxon(R.morph, R.morph_rank).pvalue))
print("  attributable to GENE SUPERVISION      = %+.4f  wins %d/8  p=%.4f"
      % (gene_part.mean(), (gene_part > 0).sum(), wilcoxon(R.ridge_shuf, R.ridge).pvalue))
print()
print("  KEY TEST: morphology truncated to ridge's rank  vs  ridge itself")
d = R.morph_rank - R.ridge
print("    diff = %+.4f  wins %d/8  p=%.4f" % (d.mean(), (d > 0).sum(), wilcoxon(R.morph_rank, R.ridge).pvalue))
print("    -> if ~0, rank collapse ALONE explains the loss; gene bottleneck adds nothing")
