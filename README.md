# FKPL: Federated Knowledge-Preserving Learning

Official implementation of **FKPL** (*Federated Knowledge-Preserving Learning*), a "divide-and-integrate" framework for federated learning under domain skew, proposed in:

> **Rethinking Model Aggregation for Federated Learning with Domain Skew: A Knowledge-Preserving Perspective**

Under domain skew, aggregating domain-specialized models hurts global knowledge. FKPL first trains domain-specialized **experts** within auto-discovered clusters (Stage I, MAKP), then **integrates** their knowledge by feature splicing and a Decoupled Prototype-Calibrated Retraining (DPCR) mechanism that recalibrates the global classifier on synthesized virtual features (Stage II).

## Repository Structure

```
FKPL_repo/
├── fkpl_PACS.py                 # Full FKPL pipeline on PACS (20 clients, 4 domains)
├── fkpl_officecaltech.py        # Full FKPL pipeline on Office-Caltech10 (10 clients, 4 domains)
├── model.py                     # Encoder / classifier architectures (ResNet-18 based) and model utilities
├── hierarchical_clustering.py   # Automatic client clustering from SVD feature subspaces
├── tSNE.py                      # t-SNE feature-visualization utility
├── requirements.txt             # Python dependencies
└── data/                        # Dataset root (not tracked, see below)
```

## Workflow Implemented

The two entry scripts (`fkpl_PACS.py`, `fkpl_officecaltech.py`) implement the complete pipeline end to end:

1. **Data preparation** — domain-skewed client partitioning
2. **Feature subspace computation** — truncated SVD of local features
3. **Automatic clustering** — hierarchical clustering of clients (`hierarchical_clustering.py`)
4. **Cluster-level federated training** — Stage I expert training (MAKP, 100 communication rounds)
5. **Feature splicing** — concatenate expert encoders into a global extractor
6. **DPCR calibration** — Stage II: domain prototype aggregation, virtual feature synthesis, unbiased global classifier retraining (200 epochs)
7. **Personalization** — domain-specific classifier heads per cluster

## Installation

```bash
conda create -n fkpl python=3.9 -y
conda activate fkpl
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; each full run takes roughly 3–6 hours depending on hardware.

## Datasets

Datasets are **not** included in this repository. Download them and place them under `data/`:

### PACS

1. Download from: https://github.com/thuml/PACS
2. Extract to `data/Homework3-PACS-master/PACS/`, so the layout is:

```
data/Homework3-PACS-master/PACS/
├── art_painting/
├── cartoon/
├── photo/
└── sketch/
```

### Office-Caltech10

1. Download Office-31 (http://pvcl.cs.bu.edu/) and Caltech-256
2. Arrange the 10 shared classes under `data/office_caltech_10/`:

```
data/office_caltech_10/
├── amazon/
├── caltech/
├── webcam/
└── dslr/
```

## Usage

```bash
# PACS (20 clients: art_painting x4, cartoon x5, photo x5, sketch x6)
python fkpl_PACS.py

# Office-Caltech10 (10 clients: caltech x4, amazon x3, webcam x2, dslr x1)
python fkpl_officecaltech.py
```

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

## License

MIT License
