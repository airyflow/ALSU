# ALSU — Active Learning with Surrogate backbones

Active learning over large molecular libraries using pre-trained backbone models
(GROVER, Uni-Mol, MoLFormer) as featurizers, with optional online backbone
finetuning at each AL iteration.

## Overview

The pipeline selects molecules from a pool (e.g. Enamine50k) to evaluate with
an expensive oracle (docking) by training a cheap surrogate model on the
backbone's embeddings, then applying an acquisition function to pick the next
batch. With `--finetune`, the backbone itself is updated on the growing labeled
set before each round, so representations improve alongside the surrogate.

```
Pool (SMILES)
    │
    ▼
Backbone (GROVER / Uni-Mol / MoLFormer)
    │ embeddings (N × D)
    ▼
Surrogate MLP (mc_dropout or ensemble)  ←── finetune backbone each round (optional)
    │ μ, σ
    ▼
Acquisition (UCB / greedy / Thompson)
    │ top-k indices
    ▼
Oracle (docking lookup)
    │ scores
    └──► labeled set grows → repeat
```

## Repository structure

```
ALSU/
├── run_active_learning.py   # single AL experiment
├── compare_backbones.py     # grid sweep + plots
├── backbone_finetuner.py    # online backbone finetuning
├── molpal/                  # AL framework (pool, acquirer, featurizer)
├── muben/
│   ├── extract_embeddings.py    # pre-extract backbone embeddings
│   └── muben/                   # backbone models, datasets, trainer
├── models/                  # pre-trained backbone weights (not tracked)
│   ├── grover/grover_base.pt
│   ├── unimol/mol_pre_all_h_220816.pt
│   └── molformer/
├── data/
│   └── Enamine50k_scores.csv.gz    # oracle docking scores
├── molpal/libraries/
│   └── Enamine50k.csv.gz           # pool SMILES (source of truth)
└── results/
    └── embed/Enamine50k/           # pre-extracted embeddings (.npz)
```

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd ALSU

# 2. Install dependencies (recommend conda)
conda env create -f muben/environment.yml
conda activate muben

pip install -e molpal/

# 3. Download backbone weights
python muben/download_models.py        # or place manually under models/
```

## Quickstart

### Option A — Static embeddings (fast)

Pre-extract embeddings once, then run AL with a frozen backbone.

```bash
# Step 1: extract embeddings for all pool molecules
python muben/extract_embeddings.py

# Step 2: run AL
python run_active_learning.py \
    --backbone unimol \
    --uq mc_dropout \
    --acq ucb
```

### Option B — Online backbone finetuning

Finetune the backbone on the labeled set before each AL round.
No pre-extraction needed; embeddings are recomputed from the updated backbone.

```bash
python run_active_learning.py \
    --backbone unimol \
    --uq mc_dropout \
    --acq ucb \
    --finetune \
    --finetune-epochs 10 \
    --finetune-lr-backbone 1e-5 \
    --finetune-lr-head 1e-4
```

### Grid sweep across all backbones

```bash
python compare_backbones.py --n-rounds 15 --batch-size 100
```

Runs all 18 combinations (3 backbones × 2 UQ × 3 acquisitions) sequentially,
then generates comparison plots in `results/comparison/Enamine50k/`.

## Arguments

### run_active_learning.py

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `Enamine50k` | Pool dataset (`Enamine10k`, `Enamine50k`, `EnamineHTS`) |
| `--backbone` | `unimol` | Backbone model (`grover`, `unimol`, `molformer`) |
| `--uq` | `mc_dropout` | Uncertainty method (`mc_dropout`, `ensemble`) |
| `--acq` | `ucb` | Acquisition function (`ucb`, `greedy`, `thompson`) |
| `--init-size` | `200` | Random molecules to label at start |
| `--batch-size` | `100` | Molecules acquired per round |
| `--n-rounds` | `15` | AL rounds |
| `--epochs` | `80` | Surrogate MLP training epochs per round |
| `--finetune` | off | Enable online backbone finetuning |
| `--finetune-epochs` | `10` | Backbone gradient epochs per round |
| `--finetune-lr-backbone` | `1e-5` | LR for backbone (keep small) |
| `--finetune-lr-head` | `1e-4` | LR for regression head |

### compare_backbones.py

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `Enamine50k` | Dataset to sweep |
| `--n-rounds` | `15` | AL rounds per run |
| `--batch-size` | `100` | Batch size per run |
| `--plot-only` | off | Skip training, just load existing runs and plot |
| `--force` | off | Re-run even if results already exist |

## Backbone models

| Model | Type | Embedding dim | Source |
|---|---|---|---|
| GROVER | 2D molecular graph | 256 | [GROVER](https://github.com/tencent-ailab/grover) |
| Uni-Mol | 3D conformational | 512 | [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) |
| MoLFormer | 1D SMILES language | 768 | [MoLFormer](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct) |

## Uncertainty quantification

- **MC Dropout** (`mc_dropout`): single MLP with dropout active at inference; 30 stochastic forward passes → mean and std
- **Ensemble** (`ensemble`): 5 independently trained MLPs; predictions averaged

## Acquisition functions

- **UCB**: `μ + 2σ` — exploits promising regions while hedging uncertainty
- **Greedy**: `μ` — pure exploitation
- **Thompson sampling**: samples from `N(μ, σ)` — stochastic exploration

## Outputs

Each run writes to `runs/al_{dataset}_{backbone}_{uq}_{acq}[_finetuned]/`:

```
runs/al_Enamine50k_unimol_mc_dropout_ucb/
├── history.json             # per-round metrics (recall, best score, n_labeled)
├── all_explored_final.csv   # all labeled molecules, sorted by score
└── iter_1/ … iter_N/
    ├── state.json
    └── scores.pkl
```

`compare_backbones.py` additionally writes plots to
`results/comparison/{dataset}/`:

- `recall_vs_round.png` — top-1% recall per AL round
- `best_score_vs_calls.png` — best docking score vs oracle calls
- `final_recall_bar.png` — final recall by backbone (mean ± std)
- `heatmap.png` — backbone × (UQ + acq) recall heatmap

## Online finetuning details

`backbone_finetuner.py` implements `BackboneFinetuner`, used automatically
when `--finetune` is passed. At each AL round it:

1. Normalizes labeled docking scores
2. Constructs mini-batches from the pre-cached pool dataset (no re-preprocessing)
3. Runs gradient steps on backbone + a lightweight `Linear(D, 1)` head
4. Re-extracts embeddings for the full pool using the updated backbone
5. Trains the surrogate MLP on the fresh embeddings

Two learning rates prevent catastrophic forgetting: `lr_backbone` (1e-5) for
the pretrained encoder, `lr_head` (1e-4) for the task head.

## Citation / acknowledgements

This project builds on:
- [MolPAL](https://github.com/coleygroup/molpal) — active learning framework
- [MUBen](https://github.com/Yinghao-Li/MUBen) — molecular uncertainty benchmarking
