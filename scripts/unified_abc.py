#!/usr/bin/env python3
"""Main three-path comparison under one clustering protocol (Fig. 1).

  Path A  predicted expression from each of the 11 published methods
  Path B  morphology-only baseline: Phikon embeddings of H&E patches
  Path C  measured expression restricted to the same 785 genes

All three paths are standardised, reduced to 50 principal components and
clustered with K-means (K = number of annotated classes, 20 seeds).
ARI is computed against the pathologist annotation.

Inputs:  her2st/  and  her2st_cluster_11.rds  (see README)
Output:  unified_abc.csv
"""
import argparse, gzip, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ANNOT = ["A1","B1","C1","D1","E1","F1","G2","H1"]

def cluster_ari(X, y, K, n_pcs=50, seeds=range(20)):
    """Shared protocol: standardise -> PCA -> KMeans -> ARI"""
    X = np.asarray(X, dtype=float)
    X = X[:, np.nanstd(X, axis=0) > 1e-12]          # drop constant columns
    X = np.nan_to_num(X)
    Xs = StandardScaler().fit_transform(X)
    p = min(n_pcs, Xs.shape[1], Xs.shape[0]-1)
    Z = PCA(n_components=p, random_state=0).fit_transform(Xs)
    a = [adjusted_rand_score(y, KMeans(K, n_init=10, random_state=s).fit_predict(Z)) for s in seeds]
    return float(np.mean(a)), float(np.std(a))

def morph_feats(root, sample, coords, patch_px=224, model="owkin/phikon", bs=32):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    Image.MAX_IMAGE_PIXELS = None
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if not hasattr(morph_feats, "_m"):
        morph_feats._p = AutoImageProcessor.from_pretrained(model)
        morph_feats._m = AutoModel.from_pretrained(model).to(dev).eval()
    proc, net = morph_feats._p, morph_feats._m
    letter = sample[0]
    img_path = next(Path(root).glob(f"data/ST-imgs/{letter}/{sample}/*.jpg"))
    img = Image.open(img_path).convert("RGB"); W,H = img.size
    half = patch_px//2; patches=[]
    for px,py in coords:
        l = max(0, min(int(round(px))-half, W-patch_px)); t = max(0, min(int(round(py))-half, H-patch_px))
        patches.append(img.crop((l,t,l+patch_px,t+patch_px)))
    out=[]
    with torch.inference_mode():
        for i in range(0,len(patches),bs):
            x = proc(images=patches[i:i+bs], return_tensors="pt").to(dev)
            out.append(net(**x).last_hidden_state[:,0,:].float().cpu().numpy())
    return np.vstack(out)

def load_real_expr(root, sample, genes):
    """Measured expression restricted to the 785 predicted genes"""
    root = Path(root)
    cp = next(root.glob(f"data/ST-cnts/{sample}.tsv.gz"))
    with gzip.open(cp,"rt") as f:
        cnt = pd.read_csv(f, sep="\t", index_col=0)
    keep = [g for g in genes if g in cnt.columns]
    X = cnt[keep].to_numpy(float)
    lib = X.sum(1, keepdims=True); lib[lib==0]=1
    return pd.DataFrame(np.log1p(X/lib*1e4), index=cnt.index, columns=keep), len(keep)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rds", default="her2st_cluster_11.rds")
    ap.add_argument("--data-root", default="her2st")
    ap.add_argument("--cache", default="morph_cache.npz")
    ap.add_argument("--out", default="unified_abc.csv")
    args = ap.parse_args()

    import pyreadr
    print("[load] reading prediction archive ...")
    df = pyreadr.read_r(args.rds)[None]
    meta = ["model_id","img_id","patch_id","label","cluster","gt_cluster","cluster_observed"]
    genes = [c for c in df.columns if c not in meta]
    print(f"[load] {df.shape}, {len(genes)} genes, {df.model_id.nunique()} methods")

    # ---- morphology features (cached) ----
    cache = {}
    if Path(args.cache).exists():
        cache = dict(np.load(args.cache, allow_pickle=True))
        print(f"[cache] loaded embeddings for {len(cache)} sections")

    rows=[]
    for img in ANNOT:
        sub = df[df.img_id==img]
        if len(sub)==0: continue
        ref = sub[sub.model_id==sub.model_id.iloc[0]].drop_duplicates("patch_id").reset_index(drop=True)
        y = ref["gt_cluster"].astype(str).values
        K = len(np.unique(y))
        spot_ids = ref["patch_id"].str.split("_").str[1].values

        # Path C: measured expression (same 785 genes)
        expr, ng = load_real_expr(args.data_root, img, genes)
        common = [s for s in spot_ids if s in expr.index]
        idx = np.array([list(spot_ids).index(s) for s in common])
        ariC, sdC = cluster_ari(expr.loc[common].to_numpy(), y[idx], K)

        # Path B: morphology features
        if img in cache:
            M = cache[img]
        else:
            sel = pd.read_csv(next(Path(args.data_root).glob(f"data/ST-spotfiles/{img}_selection.tsv")), sep="\t")
            sel["sid"] = sel.x.round().astype(int).astype(str)+"x"+sel.y.round().astype(int).astype(str)
            sel = sel.drop_duplicates("sid").set_index("sid")
            pxc = [c for c in sel.columns if c.lower() in ("pixel_x","px")][0]
            pyc = [c for c in sel.columns if c.lower() in ("pixel_y","py")][0]
            ok = [s for s in spot_ids if s in sel.index]
            coords = sel.loc[ok, [pxc,pyc]].to_numpy(float)
            F = morph_feats(args.data_root, img, coords)
            M = np.full((len(spot_ids), F.shape[1]), np.nan)
            pos = {s:i for i,s in enumerate(spot_ids)}
            for j,s in enumerate(ok): M[pos[s]] = F[j]
            cache[img] = M
            np.savez_compressed(args.cache, **cache)
            print(f"[morph] {img} features extracted {F.shape}")
        mok = ~np.isnan(M).any(1)
        ariB, sdB = cluster_ari(M[mok], y[mok], K)

        # Path A: predicted expression from each method
        pos = {p:i for i,p in enumerate(ref.patch_id)}
        for mod, g in sub.groupby("model_id"):
            g = g.drop_duplicates("patch_id").set_index("patch_id")
            keep = [p for p in ref.patch_id if p in g.index]
            ii = np.array([pos[p] for p in keep])
            ariA, sdA = cluster_ari(g.loc[keep, genes].to_numpy(), y[ii], K)
            rows.append(dict(img_id=img, model_id=mod, K=K, n=len(keep),
                             ari_pred=ariA, ari_morph=ariB, ari_realst=ariC,
                             sd_pred=sdA, sd_morph=sdB, sd_realst=sdC))
        print(f"[done] {img}: K={K} n={len(y)}  B={ariB:.4f}  C={ariC:.4f}")

    R = pd.DataFrame(rows); R["delta_BA"]=R.ari_morph-R.ari_pred; R["delta_AC"]=R.ari_pred-R.ari_realst
    R.to_csv(args.out, index=False)

    print("\n"+"="*72); print("PER-METHOD SUMMARY"); print("="*72)
    pm = R.groupby("model_id").agg(ari_pred=("ari_pred","mean"), ari_morph=("ari_morph","mean"),
        ari_realst=("ari_realst","mean"), delta_BA=("delta_BA","mean"),
        wins=("delta_BA", lambda s:int((s>0).sum()))).sort_values("ari_pred", ascending=False)
    print(pm.round(4).to_string())

    from scipy.stats import wilcoxon
    print("\n"+"="*72); print("OVERALL"); print("="*72)
    print(f"  A predicted  mean ARI = {R.ari_pred.mean():.4f}")
    print(f"  B morphology mean ARI = {R.ari_morph.mean():.4f}")
    print(f"  C measured   mean ARI = {R.ari_realst.mean():.4f}")
    print(f"\n  B>A: {(R.delta_BA>0).sum()}/{len(R)} ({(R.delta_BA>0).mean()*100:.1f}%)  mean {R.delta_BA.mean():+.4f}")
    w=wilcoxon(R.ari_morph,R.ari_pred); print(f"  Wilcoxon B vs A: p={w.pvalue:.3e}")
    print(f"\n  A>C: {(R.delta_AC>0).sum()}/{len(R)} ({(R.delta_AC>0).mean()*100:.1f}%)  mean {R.delta_AC.mean():+.4f}")
    w2=wilcoxon(R.ari_pred,R.ari_realst); print(f"  Wilcoxon A vs C: p={w2.pvalue:.3e}")
    perimg = R.groupby("img_id").first()[["ari_morph","ari_realst"]]
    w3=wilcoxon(perimg.ari_morph, perimg.ari_realst)
    print(f"\n  B vs C (per section, n=8): morphology wins {(perimg.ari_morph>perimg.ari_realst).sum()}/8  p={w3.pvalue:.4f}")
    print(f"\nsaved {args.out}")

if __name__=="__main__": main()
