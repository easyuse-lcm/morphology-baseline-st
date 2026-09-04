# Morphology-only baselines for histology-based spatial transcriptomics prediction

Code and intermediate results accompanying:

> Chenming Li. Morphology-only baselines are missing from tissue-domain
> evaluations of histology-based spatial transcriptomics prediction.
> bioRxiv (2026). doi: [TO ADD]

## What this repository shows

Methods that predict spatial gene expression from H&E images are commonly
validated by clustering the predictions into tissue domains and scoring the
result against a pathologist's annotation. This repository adds the control
missing from that analysis: clustering the image features directly, with no
prediction step.

On the eight annotated HER2-positive breast cancer sections of the her2st
dataset, that baseline (mean ARI 0.245) outperforms all eleven methods
benchmarked by Wang et al. (Nat Commun 2025; mean ARI 0.126). No model was
re-trained; all predictions are taken from the authors' public Zenodo archive.

## Setup

Python 3.11. Tested on macOS (Apple Silicon, MPS) and Linux (CPU).

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data (not included; see licences below)

1. **her2st** — `git clone https://github.com/almaan/her2st.git` into the
   repository root.
2. **Predictions** — download `her2st_cluster_11.rds` from
   https://doi.org/10.5281/zenodo.14602489 (path inside the archive:
   `Benchmarking-HE-SGE-pipeline/data/processed/her2st/`) and place it in
   the repository root.
3. Encoders download automatically on first run: Phikon from HuggingFace
   (`owkin/phikon`); DenseNet-121 and ResNet-50 from torchvision. If you are
   behind a restricted network, set `HF_ENDPOINT` to a mirror.

## Running the analysis

Scripts are in `scripts/`; run them from the repository root in this order.

| Script | Output | Manuscript |
|---|---|---|
| `verify.py` | integrity checks (stdout) | Methods |
| `unified_abc.py` | `unified_abc.csv`, `morph_cache.npz` | Fig. 1; R1–R3 |
| `exp_protocol_check.py` | `protocol_check*.csv` | Fig. 2b |
| `exp_encoder.py` | `encoder_ablation.csv` | Fig. 2a |
| `exp_sensitivity.py` | `sensitivity.csv` | Fig. 2b |
| `exp_oracle.py` | `oracle_test.csv` | Fig. 2c |
| `exp_nonmorph_ref.py` | `nonmorph_reference.csv` | Fig. 3a |
| `exp_d1d2.py` | `d2_variance.csv`, `d1d2_per_method.csv` | Fig. 3b,c |
| `exp_alpha.py`, `exp_rankmatch.py`, `exp_permnull.py` | mechanism analysis | Discussion |
| `make_figure.py` | `figure_main.png` | Fig. 1 |

All outputs are provided in `results/`, so figures can be regenerated
without re-running feature extraction.

Feature extraction for eight sections takes about five minutes on an M3
MacBook; all downstream analyses are CPU-only and finish within an hour.
`exp_permnull.py` is the slowest (1,600 ridge fits; ~20 min).

## Data provenance and licences

- her2st: Andersson et al., *Nat Commun* 12, 6012 (2021) — see that
  repository for terms.
- Predicted expression: Wang et al., *Nat Commun* 16, 1544 (2025); Zenodo
  10.5281/zenodo.14602489, CC-BY-4.0.
- Phikon: Owkin; see the model card on HuggingFace.
- Code in this repository: MIT.

## Citation

```bibtex
@article{li2026morphology,
  title   = {Morphology-only baselines are missing from tissue-domain evaluations of histology-based spatial transcriptomics prediction},
  author  = {Chenming Li},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {[TO ADD]}
}
```

## Contact

chenming.24@intl.zju.edu.cn
