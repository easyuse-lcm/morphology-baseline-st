#!/usr/bin/env python3
"""Sensitivity to clustering algorithm and annotation scope (Fig. 2b).

Recomputes the comparison with a Gaussian mixture model and with Leiden
clustering in place of K-means, and with the 'undetermined' annotation
class removed.

Output: sensitivity.csv
"""
import gzip, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
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


def ari_kmeans(X, y, K, seeds=range(20)):
    Z = embed(X)
    return float(np.mean([adjusted_rand_score(y, KMeans(K, n_init=10, random_state=s).fit_predict(Z))
                          for s in seeds]))


def ari_gmm(X, y, K, seeds=range(5)):
    Z = embed(X)
    out = []
    for s in seeds:
        try:
            lab = GaussianMixture(K, covariance_type="diag", random_state=s,
                                  reg_covar=1e-4, max_iter=200).fit_predict(Z)
            out.append(adjusted_rand_score(y, lab))
        except Exception:
            pass
    return float(np.mean(out)) if out else np.nan


def ari_leiden(X, y, K, seeds=range(3)):
    """Leiden at the resolution whose cluster count is closest to K."""
    try:
        import igraph as ig, leidenalg
    except ImportError:
        return np.nan
    Z = embed(X)
    A = kneighbors_graph(Z, n_neighbors=15, mode="connectivity", include_self=False)
    A = ((A + A.T) > 0).astype(int).tocoo()
    g = ig.Graph(n=Z.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())), directed=False)
    g.simplify()
    best, best_gap = [], 1e9
    for res in np.arange(0.1, 2.05, 0.1):
        labs = []
        for s in seeds:
            p = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                         resolution_parameter=float(res), seed=s)
            labs.append(np.array(p.membership))
        k_mean = np.mean([len(np.unique(l)) for l in labs])
        if abs(k_mean - K) < best_gap:
            best_gap, best = abs(k_mean - K), labs
    if not best:
        return np.nan
    return float(np.mean([adjusted_rand_score(y, l) for l in best]))


df = pyreadr.read_r("her2st_cluster_11.rds")[None]
meta = ["model_id","img_id","patch_id","label","cluster","gt_cluster","cluster_observed"]
genes = [c for c in df.columns if c not in meta]
cache = dict(np.load("morph_cache.npz", allow_pickle=True))

rows = []
for img in ANNOT:
    sub = df[df.img_id == img]
    ref = sub[sub.model_id == sub.model_id.iloc[0]].drop_duplicates("patch_id").reset_index(drop=True)
    y_all = ref.gt_cluster.astype(str).values
    lab_all = ref.label.astype(str).values
    sid = ref.patch_id.str.split("_").str[1].values

    with gzip.open(next(ROOT.glob("data/ST-cnts/%s.tsv.gz" % img)), "rt") as f:
        cnt = pd.read_csv(f, sep="\t", index_col=0)
    cnt = cnt.rename(columns={c: c.replace("-", ".") for c in cnt.columns})
    keep = [g for g in genes if g in cnt.columns]
    Xc = cnt[keep].to_numpy(float); lib = Xc.sum(1, keepdims=True); lib[lib == 0] = 1
    E = pd.DataFrame(np.log1p(Xc / lib * 1e4), index=cnt.index, columns=keep)

    M_all = cache[img]
    base_ok = (~np.isnan(M_all).any(1)) & np.array([s in E.index for s in sid])

    for scope in ["all", "no_undet"]:
        ok = base_ok & (lab_all != "undetermined") if scope == "no_undet" else base_ok
        y = y_all[ok]; K = len(np.unique(y))
        M = M_all[ok]
        Em = E.loc[[s for s, o in zip(sid, ok) if o]].to_numpy()
        pos = {p: i for i, p in enumerate(ref.patch_id[ok])}

        r = dict(img_id=img, scope=scope, n=len(y), K=K,
                 morph_km=ari_kmeans(M, y, K),
                 realst_km=ari_kmeans(Em, y, K),
                 morph_gmm=ari_gmm(M, y, K),
                 realst_gmm=ari_gmm(Em, y, K),
                 morph_leiden=ari_leiden(M, y, K),
                 realst_leiden=ari_leiden(Em, y, K))

        pk, pg, pl = [], [], []
        for mod, g in sub.groupby("model_id"):
            g = g.drop_duplicates("patch_id").set_index("patch_id")
            kp = [p for p in ref.patch_id[ok] if p in g.index]
            ii = np.array([pos[p] for p in kp])
            P = g.loc[kp, genes].to_numpy()
            pk.append(ari_kmeans(P, y[ii], K))
            pg.append(ari_gmm(P, y[ii], K))
            pl.append(ari_leiden(P, y[ii], K))
        r["pred_km"] = np.mean(pk); r["pred_gmm"] = np.nanmean(pg); r["pred_leiden"] = np.nanmean(pl)
        r["pred_km_wins"] = int(np.sum(np.array(pk) > r["morph_km"]))
        rows.append(r)
        print("  %-4s %-9s n=%3d K=%d | km: M %.4f P %.4f C %.4f | gmm: M %.4f P %.4f | leiden: M %.4f P %.4f"
              % (img, scope, len(y), K, r["morph_km"], r["pred_km"], r["realst_km"],
                 r["morph_gmm"], r["pred_gmm"], r["morph_leiden"], r["pred_leiden"]), flush=True)

R = pd.DataFrame(rows); R.to_csv("sensitivity.csv", index=False)

for scope in ["all", "no_undet"]:
    S = R[R.scope == scope]
    print("\n" + "=" * 78)
    print("SCOPE: %s   (n spots total = %d)" % (scope, S.n.sum()))
    print("=" * 78)
    for algo in ["km", "gmm", "leiden"]:
        m, p, c = S["morph_" + algo], S["pred_" + algo], S["realst_" + algo]
        if m.isna().all() or p.isna().all():
            print("  %-7s unavailable" % algo); continue
        d = m - p
        try:
            pv = wilcoxon(m, p).pvalue
        except Exception:
            pv = np.nan
        print("  %-7s morph %.4f  pred %.4f  realST %.4f | morph-pred %+.4f  wins %d/8  p=%.4f"
              % (algo, m.mean(), p.mean(), c.mean(), d.mean(), (d > 0).sum(), pv))
    print("  per-slide methods beating morphology (K-means): %s" % list(S.pred_km_wins))
