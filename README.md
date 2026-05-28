# ALSU — Active Learning with Surrogate and Uncertainty

Active learning for virtual screening over large molecular libraries.
Uses pre-trained backbone models (GROVER, Uni-Mol, MoLFormer) as featurizers,
with surrogate models trained on backbone embeddings to efficiently find
top-scoring molecules with minimal oracle (docking) calls.

---

## Table of contents

1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Data preparation](#data-preparation)
6. [Workflow A — Basic AL with static embeddings](#workflow-a--basic-al-with-static-embeddings)
7. [Workflow B — Online backbone finetuning](#workflow-b--online-backbone-finetuning)
8. [Workflow C — Paper experiments (Lightweight / BigFusion)](#workflow-c--paper-experiments)
9. [Component reference](#component-reference)
10. [Outputs and report](#outputs-and-report)
11. [Troubleshooting](#troubleshooting)

---

## Overview

```
Pool of SMILES  (e.g. Enamine50k, 50 k molecules)
       │
       ▼
Backbone encoder  (GROVER / Uni-Mol / MoLFormer)
       │  embeddings  (N × D)
       ▼
Surrogate model  ──── trained on labeled embeddings + docking scores
       │  μ (predicted score),  σ (uncertainty)
       ▼
Acquisition function  (UCB / greedy / Thompson / Borda count)
       │  selects top-k unlabeled molecules
       ▼
Oracle  (docking score lookup)
       │  returns scores for selected molecules
       └──►  labeled set grows → next round
```

At each active learning round the surrogate is retrained on all labeled data
collected so far.  When backbone finetuning is enabled, the backbone weights are
also updated so that embeddings improve as more labeled data becomes available.

---

## Repository structure

```
ALSU/
│
├── run_active_learning.py   # Workflow A & B: single AL run with one backbone
├── compare_backbones.py     # Grid sweep: all backbones × UQ × acquisitions
├── backbone_finetuner.py    # Online backbone finetuning helper
│
├── losses.py                # MVE loss + soft-Spearman loss + combined loss
├── surrogates.py            # Lightweight, SingleBackbone, BigFusion surrogates
├── run_experiments.py       # Workflow C: reproduce paper figure experiments
│
├── molpal/                  # Active-learning framework (pool, acquirer, featurizer)
│   └── libraries/
│       └── Enamine50k.csv.gz          # pool SMILES (source of truth)
│
├── muben/
│   ├── extract_embeddings.py          # pre-extract backbone embeddings to disk
│   ├── download_models.py             # download pre-trained backbone weights
│   └── muben/                         # backbone models, datasets, trainer
│
├── models/                  # pre-trained backbone weights (NOT tracked by git)
│   ├── grover/
│   │   └── grover_base.pt
│   ├── unimol/
│   │   └── mol_pre_all_h_220816.pt
│   └── molformer/           # HuggingFace weights downloaded automatically
│
├── data/
│   └── Enamine50k_scores.csv.gz       # oracle docking scores
│
└── results/
    ├── embed/
    │   └── Enamine50k/
    │       ├── grover_embeddings.npz
    │       ├── molformer_embeddings.npz
    │       └── unimol_embeddings.npz
    ├── comparison/          # output of compare_backbones.py
    └── experiments/         # output of run_experiments.py
        └── Enamine50k/
            ├── recall_vs_explored.png
            ├── runtime_table.csv
            └── report.html  ← auto-generated after each run
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ (CUDA 11.8+ recommended) |
| CUDA-compatible GPU | 12 GB+ VRAM for Uni-Mol / GROVER finetuning |
| Disk space | ~8 GB for model weights + embeddings |

The code runs on CPU but embedding extraction and backbone finetuning will be
very slow (30–120 min per backbone on CPU vs 2–12 min on GPU).

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd ALSU

# 2. Create and activate the conda environment (defined in muben/)
conda env create -f muben/environment.yml
conda activate muben

# 3. Install the MolPAL package in editable mode
pip install -e molpal/

# 4. Verify the installation
python -c "import muben; import molpal; print('OK')"
```

---

## Data preparation

### Step 1 — Download backbone weights

```bash
python muben/download_models.py
```

This downloads GROVER and Uni-Mol weights into `models/`.
MoLFormer is fetched automatically from HuggingFace on first use.

Alternatively, place weights manually:
```
models/grover/grover_base.pt
models/unimol/mol_pre_all_h_220816.pt
```

### Step 2 — Pre-extract embeddings  *(required for Workflows A & C)*

Run each backbone over the entire pool once and save embeddings to disk.
This is the most time-consuming step but only needs to be done once.

```bash
python muben/extract_embeddings.py
```

Expected runtimes on a single A100 GPU:

| Backbone | Time |
|---|---|
| GROVER | ~7 min (5 min features + 2 min embeddings) |
| MoLFormer | ~6 min |
| Uni-Mol | ~12 min (includes 3D conformer generation) |
| **Total** | **~25 min** |

Output files:
```
results/embed/Enamine50k/grover_embeddings.npz
results/embed/Enamine50k/molformer_embeddings.npz
results/embed/Enamine50k/unimol_embeddings.npz
```
Each `.npz` contains two arrays: `embeddings` (N × D) and `smiles` (N,) so that
row alignment is always self-documenting.

---

## Workflow A — Basic AL with static embeddings

Use pre-extracted embeddings (frozen backbone) with a simple MLP surrogate.
Fast — the backbone is never run during the AL loop.

```bash
python run_active_learning.py \
    --backbone  unimol \
    --uq        mc_dropout \
    --acq       ucb \
    --dataset   Enamine50k \
    --init-size 200 \
    --batch-size 100 \
    --n-rounds  15 \
    --epochs    80
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--backbone` | `unimol` | Which backbone embeddings to load (`grover`, `unimol`, `molformer`) |
| `--uq` | `mc_dropout` | Surrogate uncertainty: `mc_dropout` (30 stochastic passes) or `ensemble` (5 models) |
| `--acq` | `ucb` | Acquisition: `ucb` (μ + 2σ), `greedy` (μ), `thompson` (sample N(μ,σ)) |
| `--init-size` | `200` | Randomly labeled molecules at round 0 |
| `--batch-size` | `100` | Molecules queried per round |
| `--n-rounds` | `15` | Number of AL rounds |
| `--epochs` | `80` | Surrogate training epochs per round |
| `--hidden` | `512` | MLP hidden dimension |
| `--dropout` | `0.2` | Dropout rate |
| `--seed` | `42` | Random seed |

**Grid sweep across all configurations:**

```bash
python compare_backbones.py \
    --dataset    Enamine50k \
    --n-rounds   15 \
    --batch-size 100
```

Runs all 18 combinations (3 backbones × 2 UQ methods × 3 acquisitions) and
produces comparison plots in `results/comparison/Enamine50k/`.

---

## Workflow B — Online backbone finetuning

The backbone weights are updated at every AL round using the growing labeled set.
No pre-extracted embeddings are needed; embeddings are recomputed from the
updated backbone after each finetuning step.

**Run a single finetuned experiment:**

```bash
python run_active_learning.py \
    --backbone              unimol \
    --uq                    mc_dropout \
    --acq                   ucb \
    --finetune \
    --finetune-epochs       10 \
    --finetune-lr-backbone  1e-5 \
    --finetune-lr-head      1e-4
```

**Run multiple backbones for comparison:**

```bash
# Run each backbone with finetuning (independently, can be run in any order)
python run_active_learning.py --backbone unimol    --finetune --uq mc_dropout --acq ucb
python run_active_learning.py --backbone grover    --finetune --uq mc_dropout --acq ucb
python run_active_learning.py --backbone molformer --finetune --uq mc_dropout --acq ucb

# Also run the static (non-finetuned) baselines for comparison
python run_active_learning.py --backbone unimol    --uq mc_dropout --acq ucb
python run_active_learning.py --backbone grover    --uq mc_dropout --acq ucb
python run_active_learning.py --backbone molformer --uq mc_dropout --acq ucb
```

Already-finished runs are detected and skipped automatically.

**Generate comparison plots:**

```bash
# Plots all static runs found in runs/al_{dataset}_*/ (non-finetuned only)
python compare_backbones.py --plot-only

# Or re-run static experiments and plot in one command
python compare_backbones.py
```

> **Note:** `compare_backbones.py` only covers static (non-finetuned) runs and
> produces PNG plots — not an HTML report.  Finetuned run results are saved to
> `runs/al_{dataset}_{backbone}_{uq}_{acq}_finetuned/history.json` and can be
> inspected directly.  An automated comparison between finetuned and static runs
> is not yet scripted; `run_experiments.py --plot-only` (Workflow C) covers the
> paper's model comparison with HTML report generation.

**Finetuning arguments:**

| Argument | Default | Description |
|---|---|---|
| `--finetune` | off | Enable online backbone finetuning |
| `--finetune-epochs` | `10` | Gradient steps on backbone per AL round (5–20 recommended) |
| `--finetune-lr-backbone` | `1e-5` | Learning rate for backbone parameters — keep small to prevent catastrophic forgetting of pretrained features |
| `--finetune-lr-head` | `1e-4` | Learning rate for the regression head — can be larger |

**How it works** (`backbone_finetuner.py`):

At each AL round, before training the surrogate:

1. The labeled SMILES are passed to `BackboneFinetuner.finetune()`.
2. The backbone + a lightweight `Linear(D → 1)` regression head are trained
   jointly on labeled (embedding, score) pairs using MSE loss.
3. Two optimizer parameter groups use different LRs: small for the backbone
   (avoids destroying pretrained representations), larger for the head.
4. After finetuning, `BackboneFinetuner.extract_pool_embeddings()` runs a
   forward pass over all pool molecules with the updated backbone weights.
5. The surrogate is then trained on these fresh embeddings.

Output goes to `runs/al_{dataset}_{backbone}_{uq}_{acq}_finetuned/`.

---

## Workflow C — Paper experiments

Reproduces the experiments and figure from *Virtual Screening Summary 2026-04-22*.
Requires embeddings from **all three backbones** (run `extract_embeddings.py` first).

```bash
# Run all 5 model configurations with a single seed
python run_experiments.py

# Run a specific model only
python run_experiments.py --model smallfusion_5lt

# Run with 3 seeds for error bars (matches paper figure)
python run_experiments.py --seeds 42 43 44

# Plot results without rerunning (results must already exist)
python run_experiments.py --plot-only --seeds 42 43 44

# Force re-run even if results exist
python run_experiments.py --force --seeds 42
```

**Running one model and comparing it with others:**

Already-finished runs are detected automatically via `history.json` and skipped,
so you can run models independently in any order and compare them afterwards:

```bash
# Step 1 — run the model(s) you want
python run_experiments.py --model bigfusion
python run_experiments.py --model molformer       # optional — add more baselines

# Step 2 — generate the comparison report from whatever runs exist
python run_experiments.py --plot-only
```

`--plot-only` loads all completed runs, plots them together, and writes
`results/experiments/Enamine50k/report.html`.  Missing models are skipped
gracefully, so you can compare a subset at any time.

**Generate the report after a full run:**

```bash
# After python run_experiments.py finishes, the report is written automatically.
# To regenerate it without re-running the experiments:
python run_experiments.py --plot-only
```

Open the report:
```bash
# Windows
start results\experiments\Enamine50k\report.html

# Linux / macOS
xdg-open results/experiments/Enamine50k/report.html
```

**Model configurations:**

| CLI name | Paper label | Description |
|---|---|---|
| `molformer` | Molformer | MoLFormer embeddings only, SingleBackbone MVE surrogate |
| `smallfusion_5lt` | SmallFusion(5LT) | All 3 backbone embeddings concatenated, Lightweight MVE, 5 rounds |
| `mixed_3lt_2g` | Mixed(3LT+2G) | 3 Lightweight rounds → switch to 2 GROVER-only rounds |
| `mixed_4lt_1g` | Mixed(4LT+1G) | 4 Lightweight rounds → switch to 1 GROVER-only round |
| `bigfusion` | Bigfusion | 3 independent MVE surrogates combined with Borda count |

**Experiment settings** (match the paper figure):

| Setting | Value |
|---|---|
| Dataset | Enamine50k (50,240 molecules) |
| Initial labeled set | 500 random molecules |
| Batch size per round | 500 molecules |
| Number of rounds | 5 |
| Top-k metric | Top-500 recall (≈ top 1%) |
| Surrogate training epochs | 50 per round |
| Acquisition function | UCB (all models except BigFusion) |
| BigFusion acquisition | Borda count |

**Mixed scheduling rationale:**

Early AL rounds have few labeled molecules — the large foundation models
(GROVER, Uni-Mol, MoLFormer) are hard to finetune effectively on small data.
The Lightweight surrogate (fast, small) performs well early.  Later rounds
accumulate enough data that switching to a single strong backbone surrogate
(GROVER) can improve recall.  `Mixed(3LT+2G)` and `Mixed(4LT+1G)` explore this
trade-off.

---

## Component reference

### `losses.py` — Loss functions

#### MVE loss (Mean Variance Estimation)

The surrogate predicts both a mean μ and a variance σ² for each molecule.
The loss is the Gaussian negative log-likelihood:

```
L_MVE = (1/N) Σᵢ ½ [log(σ²ᵢ) + (μᵢ − yᵢ)² / σ²ᵢ]
```

This is equivalent to `torch.nn.GaussianNLLLoss` but written explicitly so that
the Spearman term can be added on top.  The variance output uses `softplus + 1e-6`
to guarantee positivity.

```python
from losses import mve_loss
loss = mve_loss(mu, var, y)   # all tensors shape (N,)
```

#### Soft Spearman loss

Hard rank is not differentiable.  We use the soft rank from Engilberge et al. (2019):

```
softRank(yᵢ) = 1 + Σⱼ≠ᵢ σ((yⱼ − yᵢ) / τ)
```

where σ is the sigmoid and τ is a temperature (smaller → harder).
The Spearman loss is then:

```
L_Spearman = 1 − ρ_soft
```

where ρ_soft is the Pearson correlation computed on the soft ranks.
`L_Spearman = 0` means perfect rank agreement; `= 2` means perfect inversion.

```python
from losses import spearman_loss
loss = spearman_loss(y_pred, y_true, tau=1.0)
```

#### Combined loss

Used for all surrogates in `run_experiments.py`:

```
L = L_MVE + λ · L_Spearman
```

```python
from losses import CombinedLoss
criterion = CombinedLoss(spearman_weight=0.1, tau=1.0)
loss = criterion(mu, var, y)
```

---

### `surrogates.py` — Surrogate models

All surrogates expose the same interface:

```python
surrogate.fit(X, y, epochs=50)       # X: (N, D), y: (N,) negated scores
mu, sigma = surrogate.predict(X)     # returns (N,), (N,)
```

#### `SingleBackboneMVESurrogate`

For a single backbone (Molformer, GROVER, or UniMol individual rounds in Mixed).

Architecture:

```
Input  (N, D_backbone)
  │
  ▼  Linear(D, 1024) → ReLU → BN → Dropout(0.25)
  │  Linear(1024, 512) → ReLU → BN → Dropout(0.25)
  │
  ▼  Shared latent h  (N, 512)
  │
  ├──► Head 1: mu1  = Linear(512,256) → ReLU → Dropout → Linear(256,1)
  │            var1 = Linear(512,256) → ReLU → Dropout → Linear(256,1) → softplus + 1e-6
  │
  └──► Head 2: mu2 / var2  (same structure)

Output: mean = 0.5*(mu1+mu2),  var = 0.5*(var1+var2)
```

The dual heads act as an implicit ensemble — averaging two independent
predictions reduces variance without the cost of training two separate models.

#### `LightweightMVESurrogate`

Concatenates all three backbone embeddings before the shared backbone:

```
Input: cat([x_uni (512), x_molf (768), x_grov (256)], dim=-1)  →  (N, 1536)
```

Same backbone + dual MVE head architecture as above.  No per-round backbone
finetuning — embeddings are static pre-extracted features.

```python
from surrogates import LightweightMVESurrogate
sur = LightweightMVESurrogate(in_dim=1536, spearman_weight=0.1)
sur.fit(X_fused, y)
mu, sigma = sur.predict(X_fused)
```

#### `BigFusionSurrogate`

Trains three independent `SingleBackboneMVESurrogate` models (one per backbone)
and combines their rankings via Borda count:

```
R_i^Borda = r_i^GROVER + r_i^MoLFormer + r_i^UniMol
```

where `r_i^M` is the rank of molecule i under model M (rank 1 = highest μ = best).
Molecules with the lowest Borda sum are selected.

```python
from surrogates import BigFusionSurrogate
dims = {"grover": 256, "molformer": 768, "unimol": 512}
sur  = BigFusionSurrogate(dims=dims)
sur.fit(X_concat, y)          # X_concat = cat([grover, molformer, unimol], axis=1)
mu, _ = sur.predict(X_concat) # mu = negative Borda sum (maximise to select best)
```

---

### `backbone_finetuner.py` — Online backbone finetuning

```python
from backbone_finetuner import BackboneFinetuner

finetuner = BackboneFinetuner(
    backbone     = "unimol",      # or "grover" / "molformer"
    dataset_name = "Enamine50k",
    pool_smiles  = pool_smiles,   # ordered list of all pool SMILES
    model_zoo    = Path("models"),
)

# At each AL round:
finetuner.finetune(
    labeled_smiles = labeled_smi,
    labeled_scores = labeled_sc,
    n_epochs       = 10,
    lr_backbone    = 1e-5,
    lr_head        = 1e-4,
)
new_embeddings = finetuner.extract_pool_embeddings()  # (N, D) float32
```

The pool dataset (conformers for UniMol, graphs for GROVER) is processed once
at initialization and cached to disk.  Subsequent calls in each AL round only
perform forward/backward passes — no re-preprocessing.

---

## Outputs and report

### run_experiments.py outputs

Each experiment run saves to `runs/exp_{dataset}_{model}_seed{seed}/`:

```
runs/exp_Enamine50k_smallfusion_5lt_seed42/
├── history.json          # [{round, n_labeled, best_score, recall, elapsed}, ...]
├── explored_final.csv    # all labeled molecules sorted by docking score
└── iter_1/ … iter_5/
    ├── state.json         # round-level metrics snapshot
    └── scores.pkl         # {smiles: score} dict for all labeled molecules
```

After all experiments complete, `run_experiments.py` automatically generates:

```
results/experiments/Enamine50k/
├── recall_vs_explored.png   # main comparison figure (matches paper)
├── runtime_table.csv        # cumulative wall-clock time per model per checkpoint
└── report.html              # self-contained HTML report (open in any browser)
```

### HTML report contents

The `report.html` file is fully self-contained (no external dependencies):

- **Configuration** — dataset, init size, batch size, rounds, loss function
- **Executive summary table** — final top-k recall (mean ± std) and best score per model
- **Recall vs molecules explored** — embedded comparison plot
- **Per-model details** — architecture, loss, round-by-round results table
- **Runtime analysis** — wall-clock time per round per model
- **References** — citations for all methods used

Open it with:
```bash
# Linux / macOS
xdg-open results/experiments/Enamine50k/report.html

# Windows
start results/experiments/Enamine50k/report.html
```

### compare_backbones.py outputs

```
results/comparison/Enamine50k/
├── recall_vs_round.png        # top-1% recall vs AL round (3 subplots by acquisition)
├── best_score_vs_calls.png    # best docking score vs oracle calls
├── final_recall_bar.png       # final recall by backbone (mean ± std across UQ × acq)
└── heatmap.png                # backbone × (UQ + acquisition) recall heatmap
```

### run_active_learning.py outputs

```
runs/al_Enamine50k_unimol_mc_dropout_ucb[_finetuned]/
├── history.json               # [{round, n_labeled, best_score, top1pct_recall}, ...]
├── all_explored_final.csv     # labeled molecules sorted by docking score
└── iter_1/ … iter_N/
    ├── state.json
    └── scores.pkl
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'muben.dataset'`

The `muben` package lives at `ALSU/muben/muben/`, not at `ALSU/muben/`.
`backbone_finetuner.py` patches `sys.path` automatically.  If you import
muben directly from a script at the ALSU root, add:

```python
import sys
sys.path.insert(0, "muben")
import muben
```

### `CUDA initialization: forward compatibility was attempted on non supported HW`

Your CUDA driver is not compatible with the installed PyTorch CUDA version.
The code will fall back to CPU automatically.  To fix, either update the GPU
driver or reinstall PyTorch matching your driver version:
```bash
nvidia-smi   # check driver version → CUDA version supported
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### `AssertionError: Missing .../grover_base.pt`

Backbone weights are not tracked by git.  Run:
```bash
python muben/download_models.py
```
or download them manually and place under `models/`.

### `AssertionError: Missing .../grover_embeddings.npz`

Pre-extracted embeddings must be generated before running Workflow A or C:
```bash
python muben/extract_embeddings.py
```

### Pool size mismatch (`49,706 > 49,699`)

Duplicate SMILES in the library CSV.  The `run_experiments.py` and
`run_active_learning.py` deduplication step handles this automatically.

### Out-of-memory during embedding extraction

Reduce the DataLoader batch size in `muben/extract_embeddings.py`:
```python
loader = DataLoader(dataset, batch_size=64, ...)  # default 256
```

### Surrogate training very slow

If CUDA is available but not being used, check:
```python
import torch; print(torch.cuda.is_available())
```
Mixed-precision training (bfloat16 / float16) is used automatically on CUDA
and disabled on CPU.

---

## References

[1] Kim, J., Nam, J., & Ryu, S. (2024). Understanding active learning of
molecular docking and its applications. arXiv:2406.12919.

[2] Ross, J. et al. (2022). Large-scale chemical language representations
capture molecular structure and properties. Nature Machine Intelligence, 4, 1256.

[3] Zhou, G. et al. (2023). Uni-Mol: A universal 3D molecular representation
learning framework. ICLR 2023.

[4] Engilberge, M. et al. (2019). SoDeep: A sorting deep net to learn ranking
loss surrogates. CVPR 2019.

[5] This project builds on [MolPAL](https://github.com/coleygroup/molpal) and
[MUBen](https://github.com/Yinghao-Li/MUBen).
