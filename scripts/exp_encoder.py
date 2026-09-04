#!/usr/bin/env python3
"""Encoder ablation (Fig. 2a).

Repeats the morphology-only baseline with an ImageNet-pretrained
DenseNet-121 and ResNet-50 in place of the pathology foundation model,
under the identical downstream protocol.

Output: encoder_ablation.csv, imagenet_cache.npz
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


def ari(X, y, K, n_pcs=50, seeds=range(20)):
    X = np.asarray(X, float)
    X = X[:, np.nanstd(X, 0) > 1e-12]
    X = np.nan_to_num(X)
    Z = PCA(min(n_pcs, X.shape[1], X.shape[0]-1), random_state=0).fit_transform(
        StandardScaler().fit_transform(X))
    return float(np.mean([adjusted_rand_score(y, KMeans(K, n_init=10, random_state=s).fit_predict(Z))
                          for s in seeds]))


def imagenet_feats(img, coords, arch="densenet121", patch_px=224, bs=32):
    """ImageNet-pretrained CNN features, global-average-pooled."""
    import torch, torchvision
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    if arch == "densenet121":
        net = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        body = net.features
    else:
        net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        body = torch.nn.Sequential(*list(net.children())[:-1])
    body = body.to(dev).eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    p = next(ROOT.glob("data/ST-imgs/%s/%s/*.jpg" % (img[0], img)))
    im = Image.open(p).convert("RGB"); W, H = im.size
    half = patch_px // 2
    patches = []
    for px, py in coords:
        l = max(0, min(int(round(px)) - half, W - patch_px))
        t = max(0, min(int(round(py)) - half, H - patch_px))
        patches.append(np.asarray(im.crop((l, t, l + patch_px, t + patch_px)), dtype=np.float32) / 255.0)

    out = []
    with torch.inference_mode():
        for i in range(0, len(patches), bs):
            x = torch.from_numpy(np.stack(patches[i:i+bs])).permute(0, 3, 1, 2)
            x = ((x - mean) / std).to(dev)
            f = body(x)
            if arch == "densenet121":
                f = torch.nn.functional.relu(f)
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
            out.append(f.float().cpu().numpy())
    return np.vstack(out)


df = pyreadr.read_r("her2st_cluster_11.rds")[None]
meta = ["model_id","img_id","patch_id","label","cluster","gt_cluster","cluster_observed"]
genes = [c for c in df.columns if c not in meta]
phikon = dict(np.load("morph_cache.npz", allow_pickle=True))

cache_path = Path("imagenet_cache.npz")
inet = dict(np.load(cache_path, allow_pickle=True)) if cache_path.exists() else {}

rows = []
for img in ANNOT:
    sub = df[df.img_id == img]
    ref = sub[sub.model_id == sub.model_id.iloc[0]].drop_duplicates("patch_id").reset_index(drop=True)
    y_all = ref.gt_cluster.astype(str).values
    sid = ref.patch_id.str.split("_").str[1].values
    K = len(np.unique(y_all))

    # measured ST on the same 785 genes (with HLA name fix)
    with gzip.open(next(ROOT.glob("data/ST-cnts/%s.tsv.gz" % img)), "rt") as f:
        cnt = pd.read_csv(f, sep="\t", index_col=0)
    cnt = cnt.rename(columns={c: c.replace("-", ".") for c in cnt.columns})
    keep = [g for g in genes if g in cnt.columns]
    X = cnt[keep].to_numpy(float); lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
    E = pd.DataFrame(np.log1p(X / lib * 1e4), index=cnt.index, columns=keep)
    okC = np.array([s in E.index for s in sid])
    a_C = ari(E.loc[[s for s, o in zip(sid, okC) if o]].to_numpy(), y_all[okC], K)

    # spot pixel coordinates
    sel = pd.read_csv(next(ROOT.glob("data/ST-spotfiles/%s_selection.tsv" % img)), sep="\t").dropna()
    sel["sid"] = sel.x.round().astype(int).astype(str) + "x" + sel.y.round().astype(int).astype(str)
    sel = sel.drop_duplicates("sid").set_index("sid")
    pxc = [c for c in sel.columns if c.lower() in ("pixel_x", "px")][0]
    pyc = [c for c in sel.columns if c.lower() in ("pixel_y", "py")][0]
    okS = np.array([s in sel.index for s in sid])
    coords = sel.loc[[s for s, o in zip(sid, okS) if o], [pxc, pyc]].to_numpy(float)

    # Phikon (cached)
    M = phikon[img]; okP = ~np.isnan(M).any(1)
    a_phikon = ari(M[okP], y_all[okP], K)

    res = dict(img_id=img, K=K, n=len(y_all), phikon=a_phikon, realst=a_C)
    for arch, dim in [("densenet121", 1024), ("resnet50", 2048)]:
        key = "%s_%s" % (img, arch)
        if key not in inet:
            inet[key] = imagenet_feats(img, coords, arch)
            np.savez_compressed(cache_path, **inet)
            print("   [%s] %s features %s" % (img, arch, inet[key].shape), flush=True)
        res[arch] = ari(inet[key], y_all[okS], K)
    res["pred_mean"] = sub.groupby("model_id").apply(
        lambda g: ari(g.drop_duplicates("patch_id").set_index("patch_id")
                      .loc[[p for p in ref.patch_id if p in set(g.patch_id)], genes].to_numpy(),
                      y_all[[i for i, p in enumerate(ref.patch_id) if p in set(g.patch_id)]], K)).mean()
    rows.append(res)
    print("  %-4s Phikon=%.4f  DenseNet=%.4f  ResNet50=%.4f  predST=%.4f  realST=%.4f"
          % (img, res["phikon"], res["densenet121"], res["resnet50"], res["pred_mean"], res["realst"]),
          flush=True)

R = pd.DataFrame(rows); R.to_csv("encoder_ablation.csv", index=False)

print("\n" + "=" * 76)
print("ENCODER ABLATION  (identical downstream protocol)")
print("=" * 76)
print(R.round(4).to_string(index=False))
print("\n  mean ARI")
for c, n in [("phikon", "Phikon (pathology FM)"), ("densenet121", "DenseNet-121 (ImageNet)"),
             ("resnet50", "ResNet-50 (ImageNet)"), ("pred_mean", "Predicted ST (11 methods)"),
             ("realst", "Measured ST")]:
    print("    %-28s %.4f" % (n, R[c].mean()))

print("\n" + "=" * 76)
print("DOES THE BASELINE BEAT PREDICTED ST?  (per slide, n=8)")
print("=" * 76)
for c, n in [("phikon", "Phikon"), ("densenet121", "DenseNet-121"), ("resnet50", "ResNet-50")]:
    d = R[c] - R.pred_mean
    print("  %-14s vs predicted ST: diff %+.4f  wins %d/8  p=%.4f"
          % (n, d.mean(), (d > 0).sum(), wilcoxon(R[c], R.pred_mean).pvalue))
    d2 = R[c] - R.realst
    print("  %-14s vs measured  ST: diff %+.4f  wins %d/8  p=%.4f"
          % (n, d2.mean(), (d2 > 0).sum(), wilcoxon(R[c], R.realst).pvalue))

print("\n  INTERPRETATION: if ImageNet encoders LOSE and Phikon WINS, the")
print("  discrepancy with Song et al. (2024) is explained by encoder quality.")
