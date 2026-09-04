#!/usr/bin/env python3
"""Permutation null for the ridge bottleneck effect.

For each section, refits the ridge 100 times to permuted expression and
100 times to random targets, and reports where the real fit falls in each
null. Per-section results are combined with a paired test across sections.

Output: permutation_null.csv, permutation_null_draws.npz
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
from scipy.stats import wilcoxon, combine_pvalues
import pyreadr

ANNOT = ["A1","B1","C1","D1","E1","F1","G2","H1"]
ROOT = Path("her2st")
ALPHA = 1e4
N_PERM = 100
SEEDS = range(5)          # fewer KMeans seeds inside the null loop, for speed


def ari(X, y, K, n_pcs=50, seeds=SEEDS):
    X = np.asarray(X, float)
    X = X[:, np.nanstd(X, 0) > 1e-12]
    X = np.nan_to_num(X)
    Xp = StandardScaler().fit_transform(X)
    Z = PCA(min(n_pcs, Xp.shape[1], Xp.shape[0]-1), random_state=0).fit_transform(Xp)
    return float(np.mean([adjusted_rand_score(y, KMeans(K, n_init=5, random_state=s).fit_predict(Z))
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
    cnt = cnt.rename(columns={c: c.replace("-", ".") for c in cnt.columns})
    keep = [g for g in genes if g in cnt.columns]
    X = cnt[keep].to_numpy(float); lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
    E = pd.DataFrame(np.log1p(X / lib * 1e4), index=cnt.index, columns=keep)
    M = cache[img]
    ok = (~np.isnan(M).any(1)) & np.array([s in E.index for s in sid])
    D[img] = dict(M=M[ok], E=E.loc[[s for s, o in zip(sid, ok) if o]].to_numpy(),
                  y=y[ok], K=len(np.unique(y)))

rows, nulls = [], {}
for img in ANNOT:
    tr = [i for i in ANNOT if i != img]
    Xtr = np.vstack([D[i]["M"] for i in tr]); Ytr = np.vstack([D[i]["E"] for i in tr])
    sx = StandardScaler().fit(Xtr)
    Xt = sx.transform(Xtr); Xte = sx.transform(D[img]["M"])
    y, K = D[img]["y"], D[img]["K"]

    a_morph = ari(D[img]["M"], y, K)
    a_real = ari(Ridge(alpha=ALPHA).fit(Xt, Ytr).predict(Xte), y, K)

    rng = np.random.default_rng(12345)
    permA, permB = [], []
    for k in range(N_PERM):
        Yp = Ytr[rng.permutation(len(Ytr))]
        permA.append(ari(Ridge(alpha=ALPHA).fit(Xt, Yp).predict(Xte), y, K))
        Yr = rng.standard_normal(Ytr.shape)
        permB.append(ari(Ridge(alpha=ALPHA).fit(Xt, Yr).predict(Xte), y, K))
    permA = np.array(permA); permB = np.array(permB)
    nulls[img] = dict(permA=permA, permB=permB)

    # one-sided: is the REAL ridge WORSE than the null?
    pA = (np.sum(permA <= a_real) + 1) / (N_PERM + 1)
    pB = (np.sum(permB <= a_real) + 1) / (N_PERM + 1)

    rows.append(dict(img_id=img, K=K, n=len(y), morph=a_morph, ridge_real=a_real,
                     nullA_mean=permA.mean(), nullA_sd=permA.std(),
                     nullB_mean=permB.mean(), nullB_sd=permB.std(),
                     p_permA=pA, p_permB=pB))
    print("  %-4s morph=%.4f  real=%.4f | nullA %.4f+-%.4f p=%.3f | nullB %.4f+-%.4f p=%.3f"
          % (img, a_morph, a_real, permA.mean(), permA.std(), pA,
             permB.mean(), permB.std(), pB), flush=True)

R = pd.DataFrame(rows); R.to_csv("permutation_null.csv", index=False)
np.savez_compressed("permutation_null_draws.npz",
                    **{("%s_%s" % (k, t)): v[t] for k, v in nulls.items() for t in ("permA","permB")})

print("\n" + "=" * 76)
print("NULL DISTRIBUTION STABILITY  (this is what the single draw got wrong)")
print("=" * 76)
print("  permuted-target null: mean %.4f, within-slide sd %.4f (range of sd %.4f-%.4f)"
      % (R.nullA_mean.mean(), R.nullA_sd.mean(), R.nullA_sd.min(), R.nullA_sd.max()))
print("  random-target  null: mean %.4f, within-slide sd %.4f"
      % (R.nullB_mean.mean(), R.nullB_sd.mean()))
print("  -> a SINGLE draw has sd ~%.3f, comparable to the effect being measured."
      % R.nullA_sd.mean())

print("\n" + "=" * 76)
print("IS THE REAL RIDGE WORSE THAN THE NULL?")
print("=" * 76)
print("  mean real ridge      = %.4f" % R.ridge_real.mean())
print("  mean permuted null   = %.4f  (diff %+.4f)" % (R.nullA_mean.mean(), R.nullA_mean.mean()-R.ridge_real.mean()))
print("  mean random null     = %.4f  (diff %+.4f)" % (R.nullB_mean.mean(), R.nullB_mean.mean()-R.ridge_real.mean()))
print()
print("  per-slide one-sided p (permuted null):", " ".join("%.3f" % p for p in R.p_permA))
print("  slides with p<0.05: %d/8" % (R.p_permA < 0.05).sum())
fA = combine_pvalues(R.p_permA, method="fisher")
fB = combine_pvalues(R.p_permB, method="fisher")
print("  Fisher combined  permuted-null p = %.5f" % fA.pvalue)
print("  Fisher combined  random-null   p = %.5f" % fB.pvalue)
print()
wA = wilcoxon(R.nullA_mean, R.ridge_real)
wB = wilcoxon(R.nullB_mean, R.ridge_real)
print("  Wilcoxon nullA-mean vs real (n=8): diff %+.4f  wins %d/8  p=%.4f"
      % ((R.nullA_mean-R.ridge_real).mean(), (R.nullA_mean > R.ridge_real).sum(), wA.pvalue))
print("  Wilcoxon nullB-mean vs real (n=8): diff %+.4f  wins %d/8  p=%.4f"
      % ((R.nullB_mean-R.ridge_real).mean(), (R.nullB_mean > R.ridge_real).sum(), wB.pvalue))

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
print("  If real ridge sits significantly BELOW both nulls, the extra loss is")
print("  specific to being supervised by REAL gene expression.")
