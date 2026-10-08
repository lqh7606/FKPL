# DomainNet-subsetA: The DomainNet Subset Used in FKPL

FKPL is **not** run on the full DomainNet (6 domains, 345 classes, ~586K images, >18 GB
download). We build a fixed, reproducible subset — **DomainNet-subsetA** — that keeps
4 domains and 10 classes while staying around 300 MB. This document records exactly
what the subset contains and how it was derived, so the experiments in
[`fkpl_domainnet4d.py`](fkpl_domainnet4d.py) can be reproduced.

---

## 1. Source

| Item | Value |
|---|---|
| Full dataset | DomainNet (Peng et al., ICCV 2019), **original ICCV19 release** (not the later "cleaned" repackaging) |
| Domains used | `clipart`, `painting`, `real`, `sketch` (4 of the 6; `infograph` and `quickdraw` are excluded) |
| Classes in full dataset | 345 |
| Download | http://ai.bu.edu/DomainNet/ — use the **ICCV19 version** table (`clipart.zip` 1.3 G / `painting.zip` 3.7 G / `real.zip` 5.6 G / `sketch.zip` 2.5 G) |

Our source tree contains 345 class folders per domain with the following image counts,
which match the official *#Original* row of the DomainNet download table
(clipart 48,837 / painting 75,759 / real 175,327 / sketch 70,386):

| Domain | Images in our source copy | Official #Original |
|---|---|---|
| clipart | 48,833 | 48,837 |
| painting | 75,759 | 75,759 |
| real | 175,327 | 175,327 |
| sketch | 70,386 | 70,386 |

All images are `.jpg`.

> If you subset from the **cleaned** release instead (clipart 48,129 / painting 72,266 /
> real 172,947 / sketch 69,128), the class-ranking step and the sampled file lists can
> shift slightly, so the resulting subset may differ by a few images per class. The
> reported FKPL numbers were obtained from the ICCV19 release.

Before subsetting, the four downloaded domains are laid out as
`<root>/domainnet/{clipart,painting,real,sketch}/<class_name>/*.jpg` (345 class
folders per domain). The class folder names follow the DomainNet vocabulary
(e.g. `beard`, `golf_club`, `streetlight`).

## 2. Subset construction protocol ("Option A": class selection + per-class quota)

Applied to the 4-domain source tree above:

1. **Class filter** — keep only classes that have **at least 150 images in every one of
   the 4 domains** (the filter is evaluated over these 4 domains only, *not* over all
   6 DomainNet domains).
2. **Class ranking** — rank the surviving candidates by their **total image count across
   the 4 domains, descending**, and keep the top **10** classes. This mirrors the
   10-class protocol of Office-Caltech10.
3. **Per-class quota** — within each `(domain, class)` folder, keep at most **200**
   images. If a folder holds more than 200, 200 are drawn with
   `random.Random(42).sample()` over the alphabetically sorted file list, using a single
   RNG instance while visiting domains in alphabetical order (`clipart` → `painting` →
   `real` → `sketch`) and classes in alphabetical order within each domain. Folders with
   ≤ 200 images are kept in full.
4. **Copy** — images are copied (not moved) into
   `data/domainnet_subsetA/<domain>/<class>/`, i.e. the same `domain/class/image`
   protocol that PACS and Office-Caltech10 use, so `torchvision.ImageFolder` loads it
   unchanged.

Parameters: `--num-classes 10 --quota 200 --min-per-domain 150 --seed 42`.

## 3. Resulting composition

**Total: 7,914 images** (≈ 303 MB), 4 domains × 10 classes.

Per domain:

| Domain | Images | Classes |
|---|---|---|
| clipart | 1,914 | 10 |
| painting | 2,000 | 10 |
| real | 2,000 | 10 |
| sketch | 2,000 | 10 |
| **Total** | **7,914** | 10 |

Per class (`clipart` is the only domain below quota — those classes have fewer than
200 images in the source and are therefore kept in full):

| Class | clipart | painting | real | sketch | Total |
|---|---|---|---|---|---|
| beard | 156 † | 200 | 200 | 200 | 756 |
| bird | 200 | 200 | 200 | 200 | 800 |
| bread | 197 † | 200 | 200 | 200 | 797 |
| golf_club | 200 | 200 | 200 | 200 | 800 |
| spider | 161 † | 200 | 200 | 200 | 761 |
| squirrel | 200 | 200 | 200 | 200 | 800 |
| streetlight | 200 | 200 | 200 | 200 | 800 |
| submarine | 200 | 200 | 200 | 200 | 800 |
| tiger | 200 | 200 | 200 | 200 | 800 |
| whale | 200 | 200 | 200 | 200 | 800 |
| **Total** | **1,914** | **2,000** | **2,000** | **2,000** | **7,914** |

† fewer than 200 available in the source clipart domain; all images kept.

## 4. Train / test split

Deterministic, seed-independent: images of each `(domain, class)` folder are sorted and
assigned by index, `i % 10 <= 7` → **train**, otherwise **test** (i.e. an exact 8:2
split, identical to the rule used for PACS and Office-Caltech10).

| Domain | Train pool | Test pool |
|---|---|---|
| clipart | 1,532 | 382 |
| painting | 1,600 | 400 |
| real | 1,600 | 400 |
| sketch | 1,600 | 400 |
| **Total** | **6,332** | **1,582** |

Evaluation is performed on the held-out test split of every training domain; the
reported number is the mean/std over the 4 domains.

## 5. Federated client layout (domain skew)

10 clients in total. Client counts and per-client sampling ratios are derived from the
SVD principal-angle diagnosis of the subset: the more **domain-central** a domain is, the
more clients it gets and the lower its per-client sampling ratio (so that per-client data
volumes stay comparable). Centrality (lower = more central, in degrees of the SVD
subspace angles): sketch 7.01 < clipart 11.03 < painting 15.44 < real 20.78.

| Domain | Clients | Sampling ratio per client | Samples per client |
|---|---|---|---|
| sketch | 4 | 25% | 400 |
| clipart | 3 | 30% | 459 |
| painting | 2 | 50% | 800 |
| real | 1 | 100% | 1,600 |
| **Total** | **10** | — | 618 (avg), range 400–1,600 |

Every client covers all 10 classes (no label skew); the skew is purely in the domain.
Each client's samples are drawn without replacement from its domain's train pool with
`np.random.permutation`.

> **Important.** `fkpl_domainnet4d.py` falls back to a built-in layout
> (`clipart 4 / painting 3 / real 2 / sketch 1`) when no diagnosis layout file is
> present. To reproduce the setting above, pass the layout explicitly:
>
> ```bash
> python fkpl_domainnet4d.py --run_encoder_stage --data_seed 7 \
>     --client_counts 3 2 1 4 --percents 0.3 0.5 1.0 0.25
> ```
>
> (the lists follow the default domain order `clipart painting real sketch`).
> Use `--data_seed` to pin the client→domain assignment and per-client sample
> selection while varying `--seed` for initialization/augmentation.

## 6. Subset-specific hyper-parameters

| Setting | Value | Note |
|---|---|---|
| Backbone | `resnet10Encoder` (32×32 input) | shared with the other two datasets |
| Input transform | `Resize((32,32))`, `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, ImageNet normalize | eval: resize + normalize only |
| `output_dim` | 10 | number of classes |
| `hidden_dim` | 512 | |
| Encoder aggregation rounds | 100 (`--encoder_round`) | 10 local epochs each |
| Client SGD | lr 0.01, momentum 0.9, weight decay 1e-5 | |
| DPCR interpolation α | `U(0.2, 0.5)` (`--alpha_low/--alpha_high`) | same as Office-Caltech10; PACS uses U(0.3, 0.7) |
| DPCR server retrain | 200 epochs, lr 0.01, batch 64 | |
| SVD basis | `--basis_mode budget`, `--budget 10`, `--n_basis 5` | |
| Default seed / data_seed | 7 | |

## 7. Expected directory layout

```
data/
└── domainnet_subsetA/
    ├── clipart/
    │   ├── beard/  ├── bird/  ├── bread/  ├── golf_club/  ├── spider/
    │   ├── squirrel/  ├── streetlight/  ├── submarine/  ├── tiger/  └── whale/
    ├── painting/   (same 10 class folders)
    ├── real/       (same 10 class folders)
    └── sketch/     (same 10 class folders)
```

`fkpl_common_4d.DEFAULT_DATA_ROOT` already points at `./data/domainnet_subsetA/`, so the
script can be launched from the repository root without extra flags.
