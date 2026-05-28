"""
run_experiments.py
Reproduces the experiments from Virtual Screening Summary 2026_4_22.

Models compared (matching the paper figure):
  molformer         — MoLFormer embeddings, SingleBackbone MVE surrogate
  smallfusion_5lt   — All-3-backbone concat, Lightweight MVE, 5 LT rounds
  mixed_3lt_2g      — 3 Lightweight rounds then 2 GROVER rounds
  mixed_4lt_1g      — 4 Lightweight rounds then 1 GROVER round
  bigfusion         — 3 independent surrogates + Borda count acquisition

All models use combined MVE + Spearman loss.
Pre-extracted backbone embeddings must exist under results/embed/{dataset}/.

Usage
-----
# Run all experiments (sequential):
python run_experiments.py

# Run a specific model only:
python run_experiments.py --model smallfusion_5lt

# Run with multiple seeds and plot:
python run_experiments.py --seeds 42 43 44 --plot-only  (after runs exist)

# Skip training and just plot existing results:
python run_experiments.py --plot-only
"""

import argparse
import base64
import datetime
import io
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch

from surrogates import (
    SingleBackboneMVESurrogate,
    LightweightMVESurrogate,
    BigFusionSurrogate,
    EnsembleFusionSurrogate,
)

ROOT       = Path(__file__).resolve().parent
EMBED_DIR  = ROOT / "results" / "embed"
DATA_DIR   = ROOT / "data"
LIBRARY_DIR = ROOT / "molpal" / "libraries"
RUNS_DIR   = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match paper figure settings
DATASET    = "Enamine50k"
INIT_SIZE  = 500
BATCH_SIZE = 500
N_ROUNDS   = 8
TOP_K      = 500    # top-1% of 50 k ≈ 500
EPOCHS     = 150    # surrogate training epochs per round (warm-start: ~300 steps round 1, ~1500 by round 5)
SEED       = 42


# ==============================================================================
# DATA LOADING
# ==============================================================================

def _load_npz(backbone: str) -> tuple[np.ndarray, list]:
    path = EMBED_DIR / DATASET / f"{backbone}_embeddings.npz"
    assert path.exists(), (
        f"Missing {path}\n"
        f"Run:  python muben/extract_embeddings.py  first."
    )
    d = np.load(path, allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["smiles"].tolist()


def load_all_embeddings() -> tuple[dict, list]:
    """
    Load GROVER, MoLFormer, and UniMol embeddings and intersect on SMILES.
    Returns (emb_dict, aligned_smiles).

    emb_dict keys: "grover", "molformer", "unimol"
    Each value is (N, D) float32.
    """
    raw = {}
    for bb in ["grover", "molformer", "unimol"]:
        mat, sml = _load_npz(bb)
        raw[bb] = {s: mat[i] for i, s in enumerate(sml)}

    # Intersect SMILES across all three backbones
    common = set(raw["grover"]) & set(raw["molformer"]) & set(raw["unimol"])

    # Preserve library order
    lib = pd.read_csv(LIBRARY_DIR / f"{DATASET}.csv.gz")
    lib.columns = lib.columns.str.strip().str.lower()
    smi_col = next(c for c in lib.columns if "smiles" in c)
    ordered = [s for s in lib[smi_col].dropna() if s in common]

    # Deduplicate
    seen, pool_smiles = set(), []
    for s in ordered:
        if s not in seen:
            pool_smiles.append(s)
            seen.add(s)

    emb_dict = {
        bb: np.stack([raw[bb][s] for s in pool_smiles])
        for bb in ["grover", "molformer", "unimol"]
    }
    print(f"[embeddings] common pool: {len(pool_smiles):,} molecules")
    print(f"             grover={emb_dict['grover'].shape[1]}d  "
          f"molformer={emb_dict['molformer'].shape[1]}d  "
          f"unimol={emb_dict['unimol'].shape[1]}d")
    return emb_dict, pool_smiles


def load_oracle() -> dict:
    gz = DATA_DIR / f"{DATASET}_scores.csv.gz"
    assert gz.exists(), f"Missing oracle file: {gz}"
    df = pd.read_csv(gz)
    df.columns = df.columns.str.strip().str.lower()
    smi_col   = next(c for c in df.columns if "smiles" in c)
    score_col = next(c for c in df.columns if "score"  in c)
    oracle = dict(zip(df[smi_col], df[score_col]))
    print(f"[oracle] {len(oracle):,} molecules  "
          f"range [{df[score_col].min():.2f}, {df[score_col].max():.2f}] kcal/mol")
    return oracle


# ==============================================================================
# ACQUISITION FUNCTIONS
# ==============================================================================

def acq_ucb(mu, sigma, beta=2.0):   return mu + beta * sigma
def acq_greedy(mu, sigma):          return mu
def acq_thompson(mu, sigma):        return np.random.normal(mu, sigma)
def acq_borda(mu, sigma):           return mu   # BigFusion: mu IS the Borda score


def _spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ without scipy dependency."""
    x_rank = np.argsort(np.argsort(x)).astype(float)
    y_rank = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


# ==============================================================================
# ACTIVE LEARNING LOOP
# ==============================================================================

class Experiment:
    """
    Self-contained AL experiment with a configurable schedule of surrogates.

    Parameters
    ----------
    emb_dict      : {"grover": (N,D_g), "molformer": (N,D_m), "unimol": (N,D_u)}
    pool_smiles   : ordered list of N SMILES (must match emb_dict rows)
    oracle        : {smiles: docking_score}  (lower = better)
    schedule      : list of (n_rounds, surrogate, X_key) tuples, where
                    X_key is "grover" | "molformer" | "unimol" | "fused" | "bigfusion"
    acq_fn        : acquisition function (mu, sigma) -> score array
    run_dir       : output directory
    """

    _SIGN = -1.0   # negate scores so surrogate maximises

    def __init__(
        self,
        emb_dict:      dict,
        pool_smiles:   list,
        oracle:        dict,
        schedule:      list,
        acq_fn,
        run_dir:       Path,
        init_size:     int  = INIT_SIZE,
        batch_size:    int  = BATCH_SIZE,
        epochs:        int  = EPOCHS,
        top_k:         int  = TOP_K,
        diverse_batch: bool = False,
    ):
        self.emb_dict   = emb_dict
        self.smiles     = np.array(pool_smiles)
        self.oracle     = oracle
        self.schedule   = schedule   # [(n_rounds, surrogate, x_key), ...]
        self.acq_fn        = acq_fn
        self.run_dir       = run_dir
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.top_k         = top_k
        self.diverse_batch = diverse_batch
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Pre-build fused embedding matrix
        self._fused = np.concatenate(
            [emb_dict["grover"], emb_dict["molformer"], emb_dict["unimol"]], axis=1
        )
        # BigFusion: full concat (same as fused; split is done inside surrogate)
        self._bigfusion = self._fused

        # Initialise labeled set randomly
        rng      = np.random.default_rng(SEED)
        init_idx = rng.choice(len(self.smiles), init_size, replace=False)
        self.labeled_idx    = set(init_idx.tolist())
        self.labeled_scores = {
            self.smiles[i]: oracle[self.smiles[i]] for i in init_idx
        }
        print(f"[init] {init_size} random molecules  "
              f"best={self._best():.3f} kcal/mol")

    def _best(self):
        return min(self.labeled_scores.values())

    def _recall(self) -> float:
        top_k_set = set(
            s for s, _ in sorted(self.oracle.items(), key=lambda x: x[1])[: self.top_k]
        )
        return sum(1 for s in self.labeled_scores if s in top_k_set) / self.top_k

    def _diverse_acquire(self, pool_idx: np.ndarray, acq_scores: np.ndarray) -> np.ndarray:
        """
        Cluster unlabeled pool into batch_size groups (k-means on L2-normalised
        MoLFormer embeddings), then pick the highest-scoring molecule per cluster.
        Guarantees the acquired batch covers the full chemical diversity of
        high-scoring candidates rather than clustering on one scaffold.
        """
        k   = min(self.batch_size, len(pool_idx))
        emb = self.emb_dict["molformer"][pool_idx].astype(np.float32)

        # L2-normalise so k-means uses cosine-like distances
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb   = emb / norms

        km     = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                                 batch_size=min(4096, len(pool_idx)))
        labels = km.fit_predict(emb)

        selected_local = []
        for c in range(k):
            in_cluster = np.where(labels == c)[0]
            if len(in_cluster) == 0:
                continue
            best = in_cluster[np.argmax(acq_scores[in_cluster])]
            selected_local.append(best)

        # Fill any empty-cluster gaps with the next-best unselected molecules
        if len(selected_local) < self.batch_size:
            chosen    = set(selected_local)
            remaining = [i for i in np.argsort(acq_scores)[::-1] if i not in chosen]
            selected_local.extend(remaining[: self.batch_size - len(selected_local)])

        return pool_idx[np.array(selected_local)]

    def _get_X(self, x_key: str) -> np.ndarray:
        if x_key == "fused" or x_key == "bigfusion":
            return self._fused
        return self.emb_dict[x_key]

    def run(self) -> list[dict]:
        history = []
        rnd_global = 0

        for n_rounds, surrogate, x_key in self.schedule:
            X_all = self._get_X(x_key)

            for _ in range(n_rounds):
                t0 = time.perf_counter()
                rnd_global += 1

                # 1. Labeled embeddings + scores
                idx  = list(self.labeled_idx)
                X_tr = X_all[idx]
                y_tr = self._SIGN * np.array(
                    [self.labeled_scores[self.smiles[i]] for i in idx],
                    dtype=np.float32,
                )

                # 2. Fit surrogate
                surrogate.fit(X_tr, y_tr, epochs=self.epochs)

                # ── Surrogate quality diagnostics ────────────────────────────
                mu_tr, _  = surrogate.predict(X_tr)
                rho_train = _spearmanr_np(mu_tr, y_tr)

                mu_all_diag, _ = surrogate.predict(X_all)
                y_all_oracle = self._SIGN * np.array(
                    [self.oracle[s] for s in self.smiles], dtype=np.float32
                )
                rho_pool = _spearmanr_np(mu_all_diag, y_all_oracle)
                print(f"  [diag] Spearman: train={rho_train:.3f}  pool={rho_pool:.3f}")
                # ── End diagnostics ───────────────────────────────────────────

                # 3. Predict unlabeled pool
                mask     = np.ones(len(self.smiles), bool)
                for i in self.labeled_idx:
                    mask[i] = False
                pool_idx = np.where(mask)[0]
                mu, sigma = surrogate.predict(X_all[pool_idx])

                # 4. Acquire batch
                acq_scores = self.acq_fn(mu, sigma)
                if self.diverse_batch:
                    selected = self._diverse_acquire(pool_idx, acq_scores)
                else:
                    top_local = np.argsort(acq_scores)[::-1][: self.batch_size]
                    selected  = pool_idx[top_local]

                # 5. Query oracle
                for i in selected:
                    smi = self.smiles[i]
                    self.labeled_scores[smi] = self.oracle[smi]
                    self.labeled_idx.add(int(i))

                # 6. Log
                recall  = self._recall()
                elapsed = time.perf_counter() - t0
                print(f"  Round {rnd_global:02d}/{sum(r for r, *_ in self.schedule)}  "
                      f"[{x_key}]  labeled={len(self.labeled_idx):,}  "
                      f"best={self._best():.3f}  top-{self.top_k} recall={recall:.1%}  "
                      f"({elapsed:.1f}s)")

                record = dict(
                    round      = rnd_global,
                    n_labeled  = len(self.labeled_idx),
                    best_score = float(self._best()),
                    recall     = float(recall),
                    elapsed    = elapsed,
                )
                history.append(record)
                self._save_checkpoint(rnd_global, record)

        self._save_final(history)
        return history

    def _save_checkpoint(self, rnd, record):
        d = self.run_dir / f"iter_{rnd}"
        d.mkdir(exist_ok=True)
        (d / "state.json").write_text(json.dumps(record, indent=2))
        with open(d / "scores.pkl", "wb") as f:
            pickle.dump(dict(self.labeled_scores), f)

    def _save_final(self, history):
        pd.DataFrame(
            sorted(self.labeled_scores.items(), key=lambda x: x[1]),
            columns=["smiles", "score"],
        ).to_csv(self.run_dir / "explored_final.csv", index=False)
        (self.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"[saved] {self.run_dir}")


# ==============================================================================
# EXPERIMENT BUILDERS
# ==============================================================================

def _fused_dim(emb_dict):
    return sum(emb_dict[k].shape[1] for k in ["grover", "molformer", "unimol"])


def build_molformer(emb_dict):
    return [(
        N_ROUNDS,
        SingleBackboneMVESurrogate(in_dim=emb_dict["molformer"].shape[1]),
        "molformer",
    )]


def build_smallfusion_5lt(emb_dict):
    return [(
        N_ROUNDS,
        LightweightMVESurrogate(in_dim=_fused_dim(emb_dict)),
        "fused",
    )]


def build_mixed_3lt_2g(emb_dict):
    return [
        (3, LightweightMVESurrogate(in_dim=_fused_dim(emb_dict)), "fused"),
        (2, SingleBackboneMVESurrogate(in_dim=emb_dict["grover"].shape[1]), "grover"),
    ]


def build_mixed_4lt_1g(emb_dict):
    return [
        (4, LightweightMVESurrogate(in_dim=_fused_dim(emb_dict)), "fused"),
        (1, SingleBackboneMVESurrogate(in_dim=emb_dict["grover"].shape[1]), "grover"),
    ]


def build_bigfusion(emb_dict):
    dims = {k: emb_dict[k].shape[1] for k in ["grover", "molformer", "unimol"]}
    return [(
        N_ROUNDS,
        BigFusionSurrogate(dims=dims),
        "bigfusion",
    )]


def build_ensemble_fusion(emb_dict):
    dims = {k: emb_dict[k].shape[1] for k in ["grover", "molformer", "unimol"]}
    return [(
        N_ROUNDS,
        EnsembleFusionSurrogate(dims=dims),
        "bigfusion",   # same fused embedding input as bigfusion
    )]


EXPERIMENTS = {
    "molformer":        (build_molformer,        acq_ucb),
    "smallfusion_5lt":  (build_smallfusion_5lt,  acq_ucb),
    "mixed_3lt_2g":     (build_mixed_3lt_2g,     acq_ucb),
    "mixed_4lt_1g":     (build_mixed_4lt_1g,     acq_ucb),
    "bigfusion":        (build_bigfusion,         acq_borda),
    "ensemble_fusion":  (build_ensemble_fusion,   acq_greedy),
}

# Experiments that use diversity-aware batch acquisition (k-means cluster + best-per-cluster)
DIVERSE_BATCH_EXPERIMENTS = {"ensemble_fusion"}


# ==============================================================================
# RUNNER
# ==============================================================================

def run_one(name: str, emb_dict: dict, pool_smiles: list, oracle: dict, seed: int):
    build_fn, acq_fn = EXPERIMENTS[name]
    schedule = build_fn(emb_dict)
    run_dir  = RUNS_DIR / f"exp_{DATASET}_{name}_seed{seed}"

    history_path = run_dir / "history.json"
    if history_path.exists():
        print(f"[skip] {run_dir.name}  (already done)")
        return json.loads(history_path.read_text())

    print(f"\n{'─'*60}")
    print(f"  Experiment: {name}  seed={seed}")
    print(f"{'─'*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    exp = Experiment(
        emb_dict       = emb_dict,
        pool_smiles    = pool_smiles,
        oracle         = oracle,
        schedule       = schedule,
        acq_fn         = acq_fn,
        run_dir        = run_dir,
        diverse_batch  = name in DIVERSE_BATCH_EXPERIMENTS,
    )
    return exp.run()


# ==============================================================================
# PLOTTING  (matches the paper figure)
# ==============================================================================

COLORS = {
    "molformer":        "#E07B4F",
    "smallfusion_5lt":  "#5B8DD9",
    "mixed_3lt_2g":     "#9B59B6",
    "mixed_4lt_1g":     "#E74C3C",
    "bigfusion":        "#6DBF87",
    "ensemble_fusion":  "#F1C40F",
}
LABELS = {
    "molformer":        "Molformer",
    "smallfusion_5lt":  "SmallFusion(5LT)",
    "mixed_3lt_2g":     "Mixed(3LT+2G)",
    "mixed_4lt_1g":     "Mixed(4LT+1G)",
    "bigfusion":        "Bigfusion",
    "ensemble_fusion":  "EnsembleFusion (new)",
}


def load_results(models, seeds) -> pd.DataFrame:
    records = []
    for name in models:
        for seed in seeds:
            path = RUNS_DIR / f"exp_{DATASET}_{name}_seed{seed}" / "history.json"
            if not path.exists():
                continue
            for r in json.loads(path.read_text()):
                records.append(dict(model=name, seed=seed, **r))
    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    for name, grp in df.groupby("model"):
        # Aggregate over seeds
        agg = grp.groupby("n_labeled")["recall"].agg(["mean", "std"]).reset_index()
        agg["recall_pct"] = agg["mean"] * 100
        agg["std_pct"]    = agg["std"]  * 100

        ax.plot(
            agg["n_labeled"], agg["recall_pct"],
            color=COLORS.get(name, "gray"),
            marker="o", markersize=4,
            label=LABELS.get(name, name),
        )
        if len(grp["seed"].unique()) > 1:
            ax.fill_between(
                agg["n_labeled"],
                agg["recall_pct"] - agg["std_pct"],
                agg["recall_pct"] + agg["std_pct"],
                alpha=0.15, color=COLORS.get(name, "gray"),
            )

    ax.set_xlabel("Molecules explored", fontsize=12)
    ax.set_ylabel(f"EN50k Percentage of Top-{TOP_K} Scores Found", fontsize=11)
    ax.set_title(
        f"Active learning on Enamine50k — top-{TOP_K} recall",
        fontsize=13, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out = out_dir / "recall_vs_explored.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] {out}")

    # Runtime table
    rt = (
        df.groupby(["model", "n_labeled"])["elapsed"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="n_labeled", values="elapsed")
    )
    rt_path = out_dir / "runtime_table.csv"
    rt.to_csv(rt_path)
    print(f"[table] {rt_path}")

    plt.close("all")


def print_summary(df: pd.DataFrame):
    final = df[df["n_labeled"] == df["n_labeled"].max()].copy()
    summary = (
        final.groupby("model")["recall"]
        .agg(mean="mean", std="std", best="max")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print(f"\n{'='*60}")
    print(f"  Final recall (top-{TOP_K}) — {DATASET}")
    print(f"{'='*60}")
    print(f"{'Model':<22} {'Mean':>8} {'Std':>8} {'Best':>8}")
    print("─" * 52)
    for _, row in summary.iterrows():
        print(f"{row['model']:<22} {row['mean']*100:>7.1f}%"
              f" {row['std']*100:>7.1f}%  {row['best']*100:>7.1f}%")


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

_MODEL_DESCRIPTIONS = {
    "molformer": {
        "full_name":    "MoLFormer (SingleBackbone)",
        "backbone":     "MoLFormer-XL (768-d language model embeddings)",
        "surrogate":    "SingleBackboneMVESurrogate — dual MVE heads on a "
                        "Linear(768→1024)→ReLU→BN→Dropout→Linear(1024→512) backbone",
        "loss":         "CombinedLoss = MVE (Gaussian NLL) + 0.1 × Spearman",
        "acquisition":  "UCB: μ + 2σ",
        "schedule":     "5 × MoLFormer rounds",
    },
    "smallfusion_5lt": {
        "full_name":    "SmallFusion-5LT (Lightweight)",
        "backbone":     "Concatenated GROVER (256-d) + MoLFormer (768-d) + UniMol (512-d) = 1536-d",
        "surrogate":    "LightweightMVESurrogate — dual MVE heads on a "
                        "Linear(1536→1024)→ReLU→BN→Dropout→Linear(1024→512) backbone",
        "loss":         "CombinedLoss = MVE + 0.1 × Spearman",
        "acquisition":  "UCB: μ + 2σ",
        "schedule":     "5 × Lightweight (fused) rounds",
    },
    "mixed_3lt_2g": {
        "full_name":    "Mixed 3LT+2G",
        "backbone":     "Phase 1: Fused 1536-d  |  Phase 2: GROVER 256-d",
        "surrogate":    "Phase 1: LightweightMVESurrogate  |  Phase 2: SingleBackboneMVESurrogate",
        "loss":         "CombinedLoss = MVE + 0.1 × Spearman",
        "acquisition":  "UCB: μ + 2σ",
        "schedule":     "3 × Lightweight (fused) rounds, then 2 × GROVER (SingleBackbone) rounds",
    },
    "mixed_4lt_1g": {
        "full_name":    "Mixed 4LT+1G",
        "backbone":     "Phase 1: Fused 1536-d  |  Phase 2: GROVER 256-d",
        "surrogate":    "Phase 1: LightweightMVESurrogate  |  Phase 2: SingleBackboneMVESurrogate",
        "loss":         "CombinedLoss = MVE + 0.1 × Spearman",
        "acquisition":  "UCB: μ + 2σ",
        "schedule":     "4 × Lightweight (fused) rounds, then 1 × GROVER (SingleBackbone) round",
    },
    "bigfusion": {
        "full_name":    "BigFusion (Borda Count)",
        "backbone":     "3 independent backbones: GROVER (256-d), MoLFormer (768-d), UniMol (512-d)",
        "surrogate":    "BigFusionSurrogate — 3 independent SingleBackboneMVESurrogates, "
                        "predictions combined via Borda count",
        "loss":         "CombinedLoss = MVE + 0.1 × Spearman (per backbone)",
        "acquisition":  "Borda count: R_i = r_i^GROVER + r_i^MoLFormer + r_i^UniMol (lower = better)",
        "schedule":     "5 × BigFusion rounds",
    },
    "ensemble_fusion": {
        "full_name":    "EnsembleFusion (Adaptive Weighted Ensemble)",
        "backbone":     "3 independent backbones: GROVER, MoLFormer, UniMol",
        "surrogate":    "EnsembleFusionSurrogate — 3 SingleBackboneMVESurrogates with adaptive "
                        "weights proportional to each backbone's training-set Spearman ρ. "
                        "Returns real (μ, σ) where σ combines inter-model disagreement (epistemic) "
                        "and per-model MVE uncertainty (aleatoric).",
        "loss":         "CombinedLoss = MVE + 0.1 × Spearman (per backbone)",
        "acquisition":  "UCB: μ_ens + 2·σ_total  (soft combination, not hard Borda)",
        "schedule":     "5 × EnsembleFusion rounds",
    },
}

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1100px; margin: 40px auto; padding: 0 24px;
    color: #2c3e50; background: #f8f9fa;
}
h1 { color: #1a252f; border-bottom: 3px solid #3498db; padding-bottom: 8px; }
h2 { color: #2980b9; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 12px; }
h3 { color: #34495e; margin-top: 28px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th { background: #2980b9; color: white; padding: 10px 14px; text-align: left; }
td { padding: 8px 14px; border-bottom: 1px solid #dee2e6; }
tr:nth-child(even) { background: #f1f4f8; }
tr:hover { background: #e8f0fe; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 0.8em; font-weight: bold; color: white; }
.best  { background: #27ae60; }
.good  { background: #2980b9; }
.ok    { background: #e67e22; }
.cfg   { background: #f1f4f8; border: 1px solid #dee2e6; border-radius: 6px;
         padding: 16px 24px; margin: 12px 0; }
.cfg dt { font-weight: bold; color: #2980b9; float: left; width: 180px; }
.cfg dd { margin-left: 190px; margin-bottom: 6px; }
.model-card { background: white; border: 1px solid #dee2e6; border-radius: 8px;
              padding: 20px; margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
img { max-width: 100%; border-radius: 8px; margin: 12px 0; }
code { background: #f1f4f8; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
.footer { font-size: 0.85em; color: #7f8c8d; border-top: 1px solid #dee2e6;
          margin-top: 48px; padding-top: 12px; }
"""


def _fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _make_recall_plot_b64(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, grp in df.groupby("model"):
        agg = grp.groupby("n_labeled")["recall"].agg(["mean", "std"]).reset_index()
        ax.plot(
            agg["n_labeled"], agg["mean"] * 100,
            color=COLORS.get(name, "gray"),
            marker="o", markersize=4,
            label=LABELS.get(name, name),
        )
        if len(grp["seed"].unique()) > 1:
            ax.fill_between(
                agg["n_labeled"],
                (agg["mean"] - agg["std"]) * 100,
                (agg["mean"] + agg["std"]) * 100,
                alpha=0.15, color=COLORS.get(name, "gray"),
            )
    ax.set_xlabel("Molecules explored", fontsize=12)
    ax.set_ylabel(f"Top-{TOP_K} recall (%)", fontsize=11)
    ax.set_title(
        f"Active learning on Enamine50k — top-{TOP_K} recall",
        fontsize=13, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _summary_table_html(df: pd.DataFrame) -> str:
    final = df[df["n_labeled"] == df["n_labeled"].max()].copy()
    summary = (
        final.groupby("model")["recall"]
        .agg(mean="mean", std="std", best="max")
        .reset_index()
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )

    rows = []
    for rank, row in summary.iterrows():
        badge_cls = "best" if rank == 0 else ("good" if rank == 1 else "ok")
        badge     = f'<span class="badge {badge_cls}">#{rank + 1}</span>'
        rows.append(
            f"<tr><td>{badge} {LABELS.get(row['model'], row['model'])}</td>"
            f"<td>{row['mean']*100:.1f}%</td>"
            f"<td>{'±'}{row['std']*100:.1f}%</td>"
            f"<td>{row['best']*100:.1f}%</td></tr>"
        )

    return (
        "<table><thead><tr>"
        "<th>Model</th><th>Mean Recall</th><th>Std</th><th>Best Recall</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _per_model_html(df: pd.DataFrame, models: list) -> str:
    parts = []
    for name in models:
        grp = df[df["model"] == name]
        if grp.empty:
            continue

        desc = _MODEL_DESCRIPTIONS.get(name, {})
        info_rows = "".join(
            f"<dt>{k.replace('_', ' ').title()}</dt><dd>{v}</dd>"
            for k, v in desc.items() if k != "full_name"
        )

        # Round-by-round table (averaged over seeds)
        rnd_agg = (
            grp.groupby("round")[["recall", "elapsed"]]
            .agg(recall_mean=("recall", "mean"), recall_std=("recall", "std"),
                 elapsed_mean=("elapsed", "mean"))
            .reset_index()
        )
        rnd_rows = "".join(
            f"<tr><td>{int(r['round'])}</td>"
            f"<td>{r['recall_mean']*100:.1f}% ± {r['recall_std']*100:.1f}%</td>"
            f"<td>{r['elapsed_mean']:.1f}s</td></tr>"
            for _, r in rnd_agg.iterrows()
        )

        parts.append(f"""
<div class="model-card">
  <h3>{desc.get('full_name', name)}</h3>
  <dl class="cfg">{info_rows}</dl>
  <h4 style="margin-top:16px">Round-by-round results</h4>
  <table>
    <thead><tr><th>Round</th><th>Top-{TOP_K} Recall (mean ± std)</th><th>Time</th></tr></thead>
    <tbody>{rnd_rows}</tbody>
  </table>
</div>""")

    return "\n".join(parts)


def _runtime_table_html(df: pd.DataFrame) -> str:
    rt = (
        df.groupby(["model", "round"])["elapsed"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="round", values="elapsed")
    )
    rt.columns = [f"Round {c}" for c in rt.columns]
    rt.index   = [LABELS.get(m, m) for m in rt.index]
    rt["Total (s)"] = rt.sum(axis=1)

    header = "<tr><th>Model</th>" + "".join(f"<th>{c}</th>" for c in rt.columns) + "</tr>"
    body   = "".join(
        f"<tr><td>{idx}</td>"
        + "".join(f"<td>{v:.1f}s</td>" for v in row)
        + "</tr>"
        for idx, row in rt.iterrows()
    )
    return f"<table><thead>{header}</thead><tbody>{body}</tbody></table>"


def generate_report(df: pd.DataFrame, out_dir: Path):
    """
    Write a fully self-contained HTML report to out_dir/report.html.
    All plots are embedded as base64 PNGs; no external dependencies.
    """
    models  = list(df["model"].unique())
    n_seeds = df["seed"].nunique()
    ts      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    recall_img = _make_recall_plot_b64(df)
    summary_tbl = _summary_table_html(df)
    per_model   = _per_model_html(df, list(EXPERIMENTS.keys()))
    runtime_tbl = _runtime_table_html(df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Virtual Screening Report — {DATASET}</title>
<style>{_CSS}</style>
</head>
<body>

<h1>Virtual Screening Report</h1>
<p style="color:#7f8c8d">Generated {ts} &nbsp;|&nbsp; Dataset: <code>{DATASET}</code>
&nbsp;|&nbsp; {n_seeds} seed(s): {sorted(df['seed'].unique().tolist())}</p>

<!-- ── Configuration ── -->
<h2>1. Configuration</h2>
<dl class="cfg">
  <dt>Dataset</dt>       <dd>{DATASET}</dd>
  <dt>Pool size</dt>     <dd>~{df['n_labeled'].max() - BATCH_SIZE*N_ROUNDS + INIT_SIZE:,} molecules (after initial {INIT_SIZE:,})</dd>
  <dt>Initial labeled</dt><dd>{INIT_SIZE:,} randomly sampled molecules</dd>
  <dt>Batch size</dt>    <dd>{BATCH_SIZE:,} molecules acquired per round</dd>
  <dt>AL rounds</dt>     <dd>{N_ROUNDS}</dd>
  <dt>Top-K target</dt>  <dd>Top {TOP_K} docking scores ({TOP_K/500*100:.0f}% of pool)</dd>
  <dt>Surrogate epochs</dt><dd>{EPOCHS} per round</dd>
  <dt>Models compared</dt><dd>{len(models)}: {', '.join(LABELS.get(m, m) for m in models)}</dd>
  <dt>Loss function</dt> <dd>CombinedLoss = MVE + 0.1 &times; Spearman</dd>
  <dt>Acquisition (default)</dt><dd>UCB (&#946;=2); BigFusion uses Borda count</dd>
</dl>

<!-- ── Executive Summary ── -->
<h2>2. Executive Summary</h2>
{summary_tbl}
<p style="font-size:.9em;color:#7f8c8d">Recall = fraction of the true top-{TOP_K}
docking hits found among all explored molecules. Higher is better.</p>

<!-- ── Recall plot ── -->
<h2>3. Recall vs. Molecules Explored</h2>
<img src="data:image/png;base64,{recall_img}"
     alt="Recall vs molecules explored">
<p style="font-size:.9em;color:#7f8c8d">
Shaded bands show ±1 std over seeds (shown only when &gt;1 seed is available).
</p>

<!-- ── Per-model details ── -->
<h2>4. Per-Model Details</h2>
{per_model}

<!-- ── Runtime ── -->
<h2>5. Runtime Analysis</h2>
{runtime_tbl}
<p style="font-size:.9em;color:#7f8c8d">
Times are seconds per AL round, averaged over seeds.
BigFusion trains three independent surrogates per round, so it is typically
2–3&times; slower than single-backbone methods.
</p>

<!-- ── References ── -->
<h2>6. References</h2>
<ol>
  <li>Graff, D. E. et al. <em>Accelerating high-throughput virtual screening through
      molecular pool-based active learning.</em> Chem. Sci. 12 (2021).</li>
  <li>Rong, Y. et al. <em>Self-supervised graph transformer on large-scale molecular data
      (GROVER).</em> NeurIPS 2020.</li>
  <li>Ross, J. et al. <em>Large-scale chemical language representations capture
      molecular structure and properties (MoLFormer).</em> Nat. Mach. Intell. 4 (2022).</li>
  <li>Zhou, G. et al. <em>Uni-Mol: a universal 3D molecular representation learning
      framework.</em> ICLR 2023.</li>
  <li>Engilberge, M. et al. <em>SoDeep: A sorting deep net to learn ranking loss
      surrogates.</em> CVPR 2019.</li>
</ol>

<div class="footer">
  ALSU — Active Learning with Surrogate Updates &nbsp;|&nbsp;
  Report generated automatically by <code>run_experiments.py</code>
</div>

</body>
</html>"""

    out_path = out_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[report] {out_path}")
    return out_path


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="all",
        choices=["all"] + list(EXPERIMENTS),
        help="Which experiment to run (default: all)",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Random seeds (one run per seed, results averaged in plots)",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help="Skip training, load existing results and plot",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run even if history.json already exists",
    )
    return p.parse_args()


def main():
    args = parse_args()
    models = list(EXPERIMENTS) if args.model == "all" else [args.model]

    print(f"\ndevice={DEVICE}  dataset={DATASET}")
    print(f"init={INIT_SIZE}  batch={BATCH_SIZE}  rounds={N_ROUNDS}  top_k={TOP_K}")
    print(f"models={models}  seeds={args.seeds}\n")

    if not args.plot_only:
        emb_dict, pool_smiles = load_all_embeddings()
        oracle = load_oracle()
        # Restrict pool to molecules with oracle scores — re-index emb_dict together
        # so that emb_dict[bb][i] always corresponds to pool_smiles[i].
        in_oracle = np.array([s in oracle for s in pool_smiles], dtype=bool)
        keep      = np.where(in_oracle)[0]
        pool_smiles = [pool_smiles[i] for i in keep]
        emb_dict    = {bb: mat[keep] for bb, mat in emb_dict.items()}
        print(f"[pool] {len(pool_smiles):,} molecules after oracle filter "
              f"({len(keep):,} kept / {(~in_oracle).sum():,} dropped — no score)")

        # ── Diagnostics ────────────────────────────────────────────────────────
        scores_all = list(oracle.values())
        print(f"\n[diag] Oracle score range: min={min(scores_all):.3f}  max={max(scores_all):.3f}  mean={np.mean(scores_all):.3f}")
        print(f"       Scores are {'NEGATIVE (lower=better, raw docking ✓)' if max(scores_all) <= 0 else 'POSITIVE (higher=better — check _SIGN and _recall direction!)'}")

        pool_set = set(pool_smiles)
        sorted_oracle = sorted(oracle.items(), key=lambda x: x[1])
        top500_smiles = [s for s, _ in sorted_oracle[:TOP_K]]
        in_pool = sum(1 for s in top500_smiles if s in pool_set)
        print(f"[diag] True top-{TOP_K} molecules in pool: {in_pool}/{TOP_K} ({in_pool/TOP_K:.0%})")
        print(f"       Max achievable recall = {in_pool/TOP_K:.0%}")
        if in_pool < TOP_K * 0.8:
            print(f"       WARNING: {TOP_K - in_pool} top-{TOP_K} molecules are NOT in the pool!")
            print(f"       Recall ceiling is {in_pool/TOP_K:.0%}, not 100%. Fix: rerun extract_embeddings.py.")
        print()
        # ── End diagnostics ────────────────────────────────────────────────────

        for name in models:
            for seed in args.seeds:
                if args.force:
                    hist_path = RUNS_DIR / f"exp_{DATASET}_{name}_seed{seed}" / "history.json"
                    if hist_path.exists():
                        hist_path.unlink()
                run_one(name, emb_dict, pool_smiles, oracle, seed)

    df = load_results(models, args.seeds)
    if df.empty:
        print("[warn] No results found. Run without --plot-only first.")
        return

    out_dir = ROOT / "results" / "experiments" / DATASET
    plot_results(df, out_dir)
    print_summary(df)
    report_path = generate_report(df, out_dir)
    print(f"\nOpen the report:\n  Windows: start {report_path}\n  Linux:   xdg-open {report_path}")


if __name__ == "__main__":
    main()
