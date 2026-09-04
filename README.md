# morphology-baseline-st
Code and results for "Morphology-only baselines are missing from tissue-domain evaluations of histology-based spatial transcriptomics prediction". Clustering H&amp;E foundation-model embeddings directly outperforms 11 published H&amp;E-to-ST methods on 8 annotated HER2+ sections.
# Morphology-only baselines for histology-based spatial transcriptomics prediction

Code and intermediate results accompanying:

> Morphology-only baselines are missing from tissue-domain
> evaluations of histology-based spatial transcriptomics prediction.
> bioRxiv (2026). 

## What this repository shows

Methods that predict spatial gene expression from H&E are commonly
validated by clustering the predictions into tissue domains and scoring
against pathologist annotation. This repository adds the control that
has been missing from that analysis: clustering the image features
directly, with no prediction step.

On the 8 annotated HER2+ breast cancer sections of the her2st dataset,
that baseline (mean ARI 0.245) outperforms all 11 published methods
benchmarked by Wang et al. (Nat Commun 2025) (mean ARI 0.126).
No model was re-trained; all predictions are taken from the authors'
public Zenodo archive.

## Reproducing the analysis

### Requirements
- Python 3.11, macOS (Apple Silicon) or Linux
- `pip install -r requirements.txt`

### Data 
1. her2st: `git clone https://github.com/almaan/her2st.git`
2. Predictions: download `her2st_cluster_11.rds` from
   https://doi.org/10.5281/zenodo.14602489
   (path: `data/processed/her2st/`)
3. Encoders download automatically (Phikon from HuggingFace;
   DenseNet/ResNet from torchvision)

### Runtime
Feature extraction for 8 sections takes ~5 min on an M3 MacBook (MPS);
all downstream analyses are CPU-only and complete in under an hour.

## Data provenance and licences
- her2st: Andersson et al., Nat Commun 2021 — see repository licence
- Predictions: Wang et al., Nat Commun 2025, Zenodo 10.5281/zenodo.14602489, CC-BY-4.0
- Phikon: Owkin, see model card on HuggingFace

## Contact
chenming.24@intl.zju.edu.cn
