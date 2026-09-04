#!/usr/bin/env python3
"""Data-integrity checks run before any analysis.

Verifies that annotation encodings are one-to-one within each section,
that spot identifiers in the prediction archive match the count matrices,
that the eleven methods' predictions are distinct, that patch coordinates
fall inside the image, that cached embeddings are non-degenerate, and that
ARI behaves as expected on random and identical labellings.
"""
import gzip, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import pyreadr

ANNOT = ["A1","B1","C1","D1","E1","F1","G2","H1"]
ROOT = Path("her2st")
df = pyreadr.read_r("her2st_cluster_11.rds")[None]
meta = ["model_id","img_id","patch_id","label","cluster","gt_cluster","cluster_observed"]
genes = [c for c in df.columns if c not in meta]

print("#" * 78)
print("V1  gt_cluster is a 1:1 numeric encoding of the pathologist label?")
print("#" * 78)
x = df[["label","gt_cluster"]].drop_duplicates().sort_values("gt_cluster")
print(x.to_string(index=False))
n_lab = df.label.nunique(); n_gt = df.gt_cluster.nunique()
print("\n  distinct labels = %d, distinct gt_cluster = %d" % (n_lab, n_gt))
bad = df.groupby("label").gt_cluster.nunique()
print("  labels mapping to >1 gt_cluster: %d" % (bad > 1).sum())
bad2 = df.groupby("gt_cluster").label.nunique()
print("  gt_cluster mapping to >1 label:  %d" % (bad2 > 1).sum())
print("  VERDICT:", "1:1 mapping OK" if (bad > 1).sum() == 0 and (bad2 > 1).sum() == 0 else "*** NOT 1:1 ***")

print("\n" + "#" * 78)
print("V2  'undetermined' spots — are they included, and how many?")
print("#" * 78)
u = df[df.model_id == df.model_id.iloc[0]].drop_duplicates("patch_id")
print(u.label.value_counts().to_string())
print("\n  undetermined share = %.1f%%" % (100 * (u.label == "undetermined").mean()))
print("  per slide:")
for img in ANNOT:
    g = u[u.img_id == img]
    print("    %-4s n=%3d  K=%d  undetermined=%d (%.1f%%)"
          % (img, len(g), g.gt_cluster.nunique(), (g.label == "undetermined").sum(),
             100*(g.label == "undetermined").mean()))

print("\n" + "#" * 78)
print("V3  Do the 11 models actually have DIFFERENT predicted values?")
print("#" * 78)
pid = df.patch_id.iloc[0]
sl = df[df.patch_id == pid][["model_id"] + genes[:4]]
print(sl.to_string(index=False))
w = df[df.patch_id == pid][genes].to_numpy()
print("\n  pairwise-identical model rows: %d of %d pairs"
      % (sum(1 for i in range(len(w)) for j in range(i+1, len(w)) if np.allclose(w[i], w[j])),
         len(w)*(len(w)-1)//2))
print("  VERDICT:", "models differ" if not np.allclose(w[0], w[1]) else "*** IDENTICAL ***")

print("\n" + "#" * 78)
print("V4  Spot alignment: patch_id -> her2st count matrix index")
print("#" * 78)
tot_m = tot_n = 0
for img in ANNOT:
    ref = df[df.img_id == img]
    ref = ref[ref.model_id == ref.model_id.iloc[0]].drop_duplicates("patch_id")
    sid = ref.patch_id.str.split("_").str[1].values
    with gzip.open(next(ROOT.glob("data/ST-cnts/%s.tsv.gz" % img)), "rt") as f:
        cnt = pd.read_csv(f, sep="\t", index_col=0)
    m = sum(s in cnt.index for s in sid)
    tot_m += m; tot_n += len(sid)
    print("    %-4s %3d/%3d matched (%.1f%%)   count index e.g. %s"
          % (img, m, len(sid), 100*m/len(sid), list(cnt.index[:3])))
print("\n  TOTAL %d/%d = %.2f%%" % (tot_m, tot_n, 100*tot_m/tot_n))

print("\n" + "#" * 78)
print("V5  Gene overlap between prediction file and her2st counts")
print("#" * 78)
with gzip.open(next(ROOT.glob("data/ST-cnts/A1.tsv.gz")), "rt") as f:
    cnt = pd.read_csv(f, sep="\t", index_col=0)
keep = [g for g in genes if g in cnt.columns]
missing = [g for g in genes if g not in cnt.columns]
print("  prediction genes = %d, matched in counts = %d, missing = %d"
      % (len(genes), len(keep), len(missing)))
print("  missing:", missing)
print("  NOTE: names like HLA.DRA are R-mangled from HLA-DRA -> real overlap may be higher")

print("\n" + "#" * 78)
print("V6  Patch coordinates: in-bounds, not degenerate?")
print("#" * 78)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
for img in ANNOT[:3]:
    sel = pd.read_csv(next(ROOT.glob("data/ST-spotfiles/%s_selection.tsv" % img)), sep="\t")
    p = Path(next(ROOT.glob("data/ST-imgs/%s/%s/*.jpg" % (img[0], img))))
    with Image.open(p) as im:
        W, H = im.size
    px = [c for c in sel.columns if c.lower() in ("pixel_x","px")][0]
    py = [c for c in sel.columns if c.lower() in ("pixel_y","py")][0]
    xs, ys = sel[px].to_numpy(), sel[py].to_numpy()
    clamp = ((xs < 112) | (xs > W-112) | (ys < 112) | (ys > H-112)).sum()
    print("    %-4s image %dx%d   x:[%.0f,%.0f] y:[%.0f,%.0f]   spots needing clamp: %d/%d"
          % (img, W, H, xs.min(), xs.max(), ys.min(), ys.max(), clamp, len(xs)))

print("\n" + "#" * 78)
print("V7  Morphology features: degenerate? cached correctly?")
print("#" * 78)
cache = dict(np.load("morph_cache.npz", allow_pickle=True))
for img in ANNOT:
    M = cache[img]
    print("    %-4s shape=%s  NaN rows=%d  mean|x|=%.4f  sd=%.4f  dup rows=%d"
          % (img, M.shape, np.isnan(M).any(1).sum(), np.nanmean(np.abs(M)),
             np.nanstd(M), len(M) - len(np.unique(np.nan_to_num(M), axis=0))))

print("\n" + "#" * 78)
print("V8  ARI sanity: random labels should give ~0")
print("#" * 78)
from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(0)
for img in ANNOT[:4]:
    ref = df[df.img_id == img]
    ref = ref[ref.model_id == ref.model_id.iloc[0]].drop_duplicates("patch_id")
    y = ref.gt_cluster.astype(str).values; K = len(np.unique(y))
    r = np.mean([adjusted_rand_score(y, rng.integers(0, K, len(y))) for _ in range(50)])
    s = adjusted_rand_score(y, y)
    print("    %-4s random ARI = %+.5f    self ARI = %.4f" % (img, r, s))
