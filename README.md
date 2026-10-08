# FKPL: Federated Knowledge-Preserving Learning

Official implementation of **FKPL** (*Federated Knowledge-Preserving Learning*), a "divide-and-integrate" framework for federated learning under domain skew, proposed in:

> **Rethinking Model Aggregation for Federated Learning with Domain Skew: A Knowledge-Preserving Perspective**

Under domain skew, aggregating domain-specialized models hurts global knowledge. FKPL first trains domain-specialized **experts** within auto-discovered clusters (Stage I, MAKP), then **integrates** their knowledge by feature splicing and a Decoupled Prototype-Calibrated Retraining (DPCR) mechanism that recalibrates the global classifier on synthesized virtual features (Stage II).

## Repository Structure

```
FKPL_repo/
├── fkpl_PACS.py                 # Full FKPL pipeline on PACS (20 clients, 4 domains)
├── fkpl_officecaltech.py        # Full FKPL pipeline on Office-Caltech10 (10 clients, 4 domains)
├── fkpl_domainnet4d.py          # Full FKPL pipeline on DomainNet-subsetA (10 clients, 4 domains)
├── fkpl_common_4d.py            # Shared data loading / SVD / clustering utilities for the DomainNet script
├── model.py                     # Encoder / classifier architectures (ResNet-10 / ResNet-18 based) and model utilities
├── hierarchical_clustering.py   # Automatic client clustering from SVD feature subspaces
├── tSNE.py                      # t-SNE feature-visualization utility
├── DOMAINNET_SUBSET.md          # Exact composition of the DomainNet subset used in the paper
├── requirements.txt             # Python dependencies
└── data/                        # Dataset root (not tracked, see below)
```

## Workflow Implemented

The three entry scripts (`fkpl_PACS.py`, `fkpl_officecaltech.py`, `fkpl_domainnet4d.py`)
implement the complete pipeline end to end:

1. **Data preparation** — domain-skewed client partitioning
2. **Feature subspace computation** — truncated SVD of local features
3. **Automatic clustering** — hierarchical clustering of clients (`hierarchical_clustering.py`)
4. **Cluster-level federated training** — Stage I expert training (MAKP, 100 communication rounds)
5. **Feature splicing** — concatenate expert encoders into a global extractor
6. **DPCR calibration** — Stage II: domain prototype aggregation, virtual feature synthesis, unbiased global classifier retraining (200 epochs)
7. **Personalization** — domain-specific classifier heads per cluster (PACS / Office-Caltech10 scripts)

## Installation

```bash
conda create -n fkpl python=3.9 -y
conda activate fkpl
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; each full run takes roughly 3–6 hours depending on hardware.

## Datasets

Datasets are **not** included in this repository. Download them and place them under
`data/` as described below.

| Dataset | Original source | Download |
|---|---|---|
| **PACS** | Li et al., *Deeper, Broader and Artier Domain Generalization*, ICCV 2017 — 4 domains × 7 classes, 9,991 images | https://github.com/MachineLearning2020/Homework3-PACS (the `PACS/` folder) |
| **Office-Caltech10** | Office-31 (Saenko et al., ICML 2010) ∩ Caltech-256 (Griffin et al., 2007), 10 shared classes, 4 domains, 2,533 images | Aggregated download list: https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md — Caltech-256 from https://www.vision.caltech.edu/ (Datasets) |
| **DomainNet-subsetA** | Peng et al., *Moment Matching for Multi-Source Domain Adaptation*, ICCV 2019 — a fixed 4-domain / 10-class subset (7,914 images) | Official zips: http://ai.bu.edu/DomainNet/ (ICCV19 version) — subset recipe and statistics: [DOMAINNET_SUBSET.md](DOMAINNET_SUBSET.md) |

### PACS

Extract the four domain folders so the layout is:

```
data/Homework3-PACS-master/PACS/
├── art_painting/   # 2,048 images
├── cartoon/        # 2,344 images
├── photo/          # 1,670 images
└── sketch/         # 3,929 images
```

### Office-Caltech10

The 10 shared classes, using the folder names the loader expects (`back_pack`, `bike`,
`calculator`, `headphones`, `keyboard`, `laptop_computer`, `monitor`, `mouse`, `mug`,
`projector`) from the four domains:

```
data/office_caltech_10/
├── amazon/      # 958 images
├── caltech/     # 1,123 images
├── webcam/      # 295 images
└── dslr/        # 157 images
```

### DomainNet-subsetA

FKPL uses a 300 MB subset of DomainNet (**4 domains**: clipart / painting / real /
sketch, **10 classes**, 7,914 images), not the full 345-class dataset. The class
selection rule, per-class image counts, the 8:2 split and the client layout are all
documented in [DOMAINNET_SUBSET.md](DOMAINNET_SUBSET.md). Build it as:

```
data/domainnet_subsetA/
├── clipart/<class_name>/*.jpg      # 1,914 images, 10 classes
├── painting/<class_name>/*.jpg     # 2,000 images, 10 classes
├── real/<class_name>/*.jpg         # 2,000 images, 10 classes
└── sketch/<class_name>/*.jpg       # 2,000 images, 10 classes
```

## Usage

```bash
# PACS (20 clients: art_painting x4, cartoon x5, photo x5, sketch x6)
python fkpl_PACS.py

# Office-Caltech10 (10 clients: caltech x4, amazon x3, webcam x2, dslr x1)
python fkpl_officecaltech.py

# DomainNet-subsetA (10 clients: clipart x3, painting x2, real x1, sketch x4)
# The client layout must be given explicitly; see DOMAINNET_SUBSET.md section 5.
python fkpl_domainnet4d.py --run_encoder_stage --data_seed 7 \
    --client_counts 3 2 1 4 --percents 0.3 0.5 1.0 0.25
```

The PACS and Office-Caltech10 scripts are self-contained scripts with hard-coded
hyper-parameters, whereas `fkpl_domainnet4d.py` exposes its settings as command-line
arguments (`--seed`, `--data_seed`, `--encoder_round`, `--alpha_low/--alpha_high`,
`--use_hc_for_training`, …).

### Key Parameters

| Parameter | Location | Description |
|---|---|---|
| `cluster_alpha` | entry scripts | Clustering threshold; larger → fewer clusters (PACS: 4.2, Office: 5.3) |
| `selected_domain_dict` | entry scripts | Client-to-domain distribution |
| `local_epochs` | entry scripts | Local training epochs per round (default 10) |
| `uniform_left / uniform_right` | entry scripts | DPCR virtual-feature interpolation range α ~ U(a, b) (PACS: 0.3/0.7, Office: 0.2/0.5) |
| `n_epoch` | `training_global_classifier_with_mix_features` | DPCR classifier retraining epochs (default 200) |

### Output

Each run prints the clustering result, then the FKPL global model accuracy after DPCR:

```
[FKPL global model] mean test accuracy: ...
[FKPL global model] individual test accuracy, art_painting: ..., cartoon: ..., photo: ..., sketch: ...
```

Trained expert encoders and calibrated classifiers are saved as `.pth` checkpoints under the run directory.

## Citation

If you find this work useful, please cite:

```bibtex
@article{fkpl2026,
  title   = {Rethinking Model Aggregation for Federated Learning with Domain Skew: A Knowledge-Preserving Perspective},
  year    = {2026}
}
```

Datasets used in this repository are published separately; please also cite:

```bibtex
@inproceedings{li2017deeper,          title={Deeper, broader and artier domain generalization},
author={Li, Da and Yang, Yongxin and Song, Yi-Zhe and Hospedales, Timothy M.},
booktitle={IEEE ICCV}, year={2017}}

@inproceedings{saenko2010adapting,    title={Adapting algorithms for visual domains between closely related and distantly related domains},
author={Saenko, Kate and Kulis, Brian and Fritz, Mario and Darrell, Trevor},
booktitle={NeurIPS}, year={2010}}

@inproceedings{peng2019moment,        title={Moment matching for multi-source domain adaptation},
author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},
booktitle={IEEE ICCV}, year={2019}}
```

## License

MIT License
