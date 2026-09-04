#!/usr/bin/env python3
"""Comparison under a molecular reference (Fig. 3a).

Replaces the pathologist annotation with the K-means partition of the
measured expression as the reference, and repeats the three-path
comparison. The two references are also compared with each other.

Output: nonmorph_reference.csv
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


def embed(X, n_pcs=50):
    X = np.asarray(X, float)
    X = X[:, np.nanstd(X, 0) > 1e-12]
    X = np.nan_to_num(X)
    Xs = StandardScaler().fit_transform(X)
    return PCA(min(n_pcs, Xs.shape[1], Xs.shape[0]-1), random_state=0).fit_transform(Xs)


def labels(X, K, seed=0):
    return KMeans(K, n_init=10, random_state=seed).fit_predict(embed(X))


def ari_multi(X, y, K, seeds=range(20)):
    Z = embed(X)
    return float(np.mean([adjusted_rand_score(y, KMeans(K, n_init=10, random_state=s).fit_predict(Z))
                          for s in seeds]))


df = pyreadr.read_r("her2st_cluster_11.rds")[None]
meta = ["model_id","img_id","patch_id","label","cluster","gt_cluster","cluster_observed"]
genes = [c for c in df.columns if c not in meta]
phikon = dict(np.load("morph_cache.npz", allow_pickle=True))

rows = []
for img in ANNOT:
    sub = df[df.img_id == img]
    ref = sub[sub.model_id == sub.model_id.iloc[0]].drop_duplicates("patch_id").reset_index(drop=True)
    y_path = ref.gt_cluster.astype(str).values
    sid = ref.patch_id.str.split("_").str[1].values
    K = len(np.unique(y_path))

    with gzip.open(next(ROOT.glob("data/ST-cnts/%s.tsv.gz" % img)), "rt") as f:
        cnt = pd.read_csv(f, sep="\t", index_col=0)
    cnt = cnt.rename(columns={c: c.replace("-", ".") for c in cnt.columns})
    keep = [g for g in genes if g in cnt.columns]
    Xc = cnt[keep].to_numpy(float); lib = Xc.sum(1, keepdims=True); lib[lib == 0] = 1
    E = pd.DataFrame(np.log1p(Xc / lib * 1e4), index=cnt.index, columns=keep)

    M = phikon[img]
    ok = (~np.isnan(M).any(1)) & np.array([s in E.index for s in sid])
    M = M[ok]
    Emat = E.loc[[s for s, o in zip(sid, ok) if o]].to_numpy()
    y_path = y_path[ok]
    idx_ok = np.where(ok)[0]

    # GT2: molecular reference = clustering of MEASURED ST (fixed seed)
    y_mol = labels(Emat, K, seed=0).astype(str)

    r = dict(img_id=img, K=K, n=len(y_path),
             # under pathologist annotation
             morph_GT1=ari_multi(M, y_path, K),
             realst_GT1=ari_multi(Emat, y_path, K),
             # under molecular reference
             morph_GT2=ari_multi(M, y_mol, K))

    # how similar are the two references to each other?
    r["GT1_vs_GT2"] = adjusted_rand_score(y_path, y_mol)

    preds = {}
    keep_pid = set(ref.patch_id[idx_ok])
    for mod, g in sub.groupby("model_id"):
        g = g.drop_duplicates("patch_id").set_index("patch_id")
        kp = [p for p in ref.patch_id[idx_ok] if p in g.index]
        pos = {p: i for i, p in enumerate(ref.patch_id[idx_ok])}
        ii = np.array([pos[p] for p in kp])
        P = g.loc[kp, genes].to_numpy()
        preds[mod] = (ari_multi(P, y_path[ii], K), ari_multi(P, y_mol[ii], K))
    r["pred_GT1"] = np.mean([v[0] for v in preds.values()])
    r["pred_GT2"] = np.mean([v[1] for v in preds.values()])
    r["predbest_GT2"] = max(v[1] for v in preds.values())
    rows.append(r)
    print("  %-4s K=%d | GT1: morph %.4f pred %.4f realST %.4f | GT2: morph %.4f pred %.4f | GT1~GT2 ARI %.4f"
          % (img, K, r["morph_GT1"], r["pred_GT1"], r["realst_GT1"],
             r["morph_GT2"], r["pred_GT2"], r["GT1_vs_GT2"]), flush=True)

R = pd.DataFrame(rows); R.to_csv("nonmorph_reference.csv", index=False)

print("\n" + "=" * 78)
print("HOW DIFFERENT ARE THE TWO REFERENCES?")
print("=" * 78)
print("  ARI(pathologist annotation, measured-ST clustering) = %.4f" % R.GT1_vs_GT2.mean())
print("  -> low value confirms the molecular reference is genuinely different")

print("\n" + "=" * 78)
print("GT1  pathologist annotation  (morphology-derived)")
print("=" * 78)
print("  morphology   %.4f" % R.morph_GT1.mean())
print("  predicted ST %.4f" % R.pred_GT1.mean())
print("  measured ST  %.4f" % R.realst_GT1.mean())
d = R.morph_GT1 - R.pred_GT1
print("  morph - pred: %+.4f  wins %d/8  p=%.4f" % (d.mean(), (d > 0).sum(),
                                                    wilcoxon(R.morph_GT1, R.pred_GT1).pvalue))

print("\n" + "=" * 78)
print("GT2  clustering of MEASURED ST  (molecular, NOT morphology-derived)")
print("=" * 78)
print("  morphology        %.4f" % R.morph_GT2.mean())
print("  predicted ST      %.4f" % R.pred_GT2.mean())
print("  predicted ST best %.4f" % R.predbest_GT2.mean())
d2 = R.morph_GT2 - R.pred_GT2
print("  morph - pred: %+.4f  wins %d/8  p=%.4f" % (d2.mean(), (d2 > 0).sum(),
                                                    wilcoxon(R.morph_GT2, R.pred_GT2).pvalue))
d3 = R.morph_GT2 - R.predbest_GT2
print("  morph - best pred: %+.4f  wins %d/8  p=%.4f" % (d3.mean(), (d3 > 0).sum(),
                                                         wilcoxon(R.morph_GT2, R.predbest_GT2).pvalue))

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
print("  If morphology also wins under GT2, the finding is NOT an artefact of")
print("  the reference being morphology-derived.")
print("  If predicted ST wins under GT2, that is where its real value lies.")
