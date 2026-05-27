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
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch

from surrogates import (
    SingleBackboneMVESurrogate,
    LightweightMVESurrogate,
    BigFusionSurrogate,
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
N_ROUNDS   = 5
TOP_K      = 500    # top-1% of 50 k ≈ 500
EPOCHS     = 50     # surrogate training epochs per round
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
        emb_dict:    dict,
        pool_smiles: list,
        oracle:      dict,
        schedule:    list,
        acq_fn,
        run_dir:     Path,
        init_size:   int = INIT_SIZE,
        batch_size:  int = BATCH_SIZE,
        epochs:      int = EPOCHS,
        top_k:       int = TOP_K,
    ):
        self.emb_dict   = emb_dict
        self.smiles     = np.array(pool_smiles)
        self.oracle     = oracle
        self.schedule   = schedule   # [(n_rounds, surrogate, x_key), ...]
        self.acq_fn     = acq_fn
        self.run_dir    = run_dir
        self.batch_size = batch_size
        self.epochs     = epochs
        self.top_k      = top_k
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

                # 3. Predict unlabeled pool
                mask     = np.ones(len(self.smiles), bool)
                for i in self.labeled_idx:
                    mask[i] = False
                pool_idx = np.where(mask)[0]
                mu, sigma = surrogate.predict(X_all[pool_idx])

                # 4. Acquire top-k
                acq_scores = self.acq_fn(mu, sigma)
                top_local  = np.argsort(acq_scores)[::-1][: self.batch_size]
                selected   = pool_idx[top_local]

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


EXPERIMENTS = {
    "molformer":        (build_molformer,       acq_ucb),
    "smallfusion_5lt":  (build_smallfusion_5lt, acq_ucb),
    "mixed_3lt_2g":     (build_mixed_3lt_2g,    acq_ucb),
    "mixed_4lt_1g":     (build_mixed_4lt_1g,    acq_ucb),
    "bigfusion":        (build_bigfusion,        acq_borda),
}


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
        emb_dict    = emb_dict,
        pool_smiles = pool_smiles,
        oracle      = oracle,
        schedule    = schedule,
        acq_fn      = acq_fn,
        run_dir     = run_dir,
    )
    return exp.run()


# ==============================================================================
# PLOTTING  (matches the paper figure)
# ==============================================================================

COLORS = {
    "molformer":       "#E07B4F",
    "smallfusion_5lt": "#5B8DD9",
    "mixed_3lt_2g":    "#9B59B6",
    "mixed_4lt_1g":    "#E74C3C",
    "bigfusion":       "#6DBF87",
}
LABELS = {
    "molformer":       "Molformer",
    "smallfusion_5lt": "SmallFusion(5LT)",
    "mixed_3lt_2g":    "Mixed(3LT+2G)",
    "mixed_4lt_1g":    "Mixed(4LT+1G)",
    "bigfusion":       "Bigfusion",
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
        # Restrict pool to molecules with oracle scores
        pool_smiles = [s for s in pool_smiles if s in oracle]

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


if __name__ == "__main__":
    main()
