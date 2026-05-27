# compare_backbones.py
"""
Runs the AL loop for every backbone × UQ × acquisition combination
and produces comparison plots.

Usage:
    python compare_backbones.py --n-rounds 15 --batch-size 100
"""

import argparse
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT     = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"


# ==============================================================================
# 1. RUN ALL COMBINATIONS
# ==============================================================================

def run_all(args):
    import subprocess, sys

    backbones = ["unimol", "grover", "molformer"]
    uqs       = ["mc_dropout", "ensemble"]
    acqs      = ["ucb", "greedy", "thompson"]

    combos = list(itertools.product(backbones, uqs, acqs))
    print(f"Running {len(combos)} combinations × {args.n_rounds} rounds each\n")

    for backbone, uq, acq in combos:
        run_name = f"al_{args.dataset}_{backbone}_{uq}_{acq}"
        history_path = RUNS_DIR / run_name / "history.json"

        if history_path.exists() and not args.force:
            print(f"[skip] {run_name}  (already done, use --force to rerun)")
            continue

        print(f"\n{'─'*60}")
        print(f"  {backbone} | {uq} | {acq}")
        print(f"{'─'*60}")

        cmd = [
            sys.executable, "run_active_learning.py",
            "--dataset",    args.dataset,
            "--backbone",   backbone,
            "--uq",         uq,
            "--acq",        acq,
            "--n-rounds",   str(args.n_rounds),
            "--batch-size", str(args.batch_size),
            "--init-size",  str(args.init_size),
            "--epochs",     str(args.epochs),
            "--seed",       str(args.seed),
        ]
        subprocess.run(cmd, check=True)


# ==============================================================================
# 2. LOAD RESULTS
# ==============================================================================

def load_results(dataset: str) -> pd.DataFrame:
    """
    Walk runs/ and collect history.json from every matching run.
    Returns a long-form DataFrame with columns:
        backbone, uq, acq, round, n_labeled, best_score, top1pct_recall
    """
    records = []
    for history_path in RUNS_DIR.glob(f"al_{dataset}_*/history.json"):
        parts   = history_path.parent.name.split("_")
        # al_{dataset}_{backbone}_{uq}_{acq}
        # dataset may contain underscores (Enamine50k) so strip from left
        prefix  = f"al_{dataset}_"
        suffix  = history_path.parent.name[len(prefix):]   # e.g. "unimol_mc_dropout_ucb"
        backbone, uq, acq = suffix.split("_", 2)  # split on first two only
        # uq contains underscore: mc_dropout → split once more
        # safer: split on known values
        for bb in ["unimol", "grover", "molformer"]:
            if suffix.startswith(bb):
                backbone = bb
                rest = suffix[len(bb)+1:]
                break
        for u in ["mc_dropout", "ensemble"]:
            if rest.startswith(u):
                uq  = u
                acq = rest[len(u)+1:]
                break

        history = json.loads(history_path.read_text())
        for r in history:
            records.append(dict(backbone=backbone, uq=uq, acq=acq, **r))

    df = pd.DataFrame(records)
    print(f"Loaded {len(df['backbone'].unique())} backbones, "
          f"{len(df)} round-records from {RUNS_DIR}")
    return df


# ==============================================================================
# 3. PLOTS
# ==============================================================================

COLORS = {
    "unimol":    "#5B8DD9",
    "grover":    "#E07B4F",
    "molformer": "#6DBF87",
}
LINES = {
    "mc_dropout": "-",
    "ensemble":   "--",
}
MARKERS = {
    "ucb":      "o",
    "greedy":   "s",
    "thompson": "^",
}


def _label(backbone, uq, acq):
    uq_short  = {"mc_dropout": "MCD", "ensemble": "Ens"}[uq]
    acq_short = {"ucb": "UCB", "greedy": "Greedy", "thompson": "TS"}[acq]
    return f"{backbone} / {uq_short} / {acq_short}"


def plot_top1pct_recall(df: pd.DataFrame, out: Path):
    """Top-1% recall vs round — one line per (backbone × uq × acq)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    acq_list  = df["acq"].unique()

    for ax, acq in zip(axes, acq_list):
        sub = df[df["acq"] == acq]
        for (backbone, uq), grp in sub.groupby(["backbone", "uq"]):
            grp = grp.sort_values("round")
            ax.plot(grp["round"], grp["top1pct_recall"] * 100,
                    color=COLORS[backbone],
                    linestyle=LINES[uq],
                    marker=MARKERS[acq],
                    markersize=4,
                    label=_label(backbone, uq, acq))
        ax.set_title(f"Acquisition: {acq.upper()}", fontsize=12)
        ax.set_xlabel("AL round")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Top-1% recall")
    fig.suptitle("Top-1% recall vs AL round", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] {out}")


def plot_best_score(df: pd.DataFrame, out: Path):
    """Best docking score found vs molecules evaluated."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for (backbone, uq, acq), grp in df.groupby(["backbone", "uq", "acq"]):
        grp = grp.sort_values("round")
        ax.plot(grp["n_labeled"], grp["best_score"],
                color=COLORS[backbone],
                linestyle=LINES[uq],
                marker=MARKERS[acq],
                markersize=4,
                label=_label(backbone, uq, acq),
                alpha=0.85)

    ax.set_xlabel("Molecules evaluated")
    ax.set_ylabel("Best docking score (kcal/mol)")
    ax.set_title("Best score found vs oracle calls", fontsize=13)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] {out}")


def plot_final_bar(df: pd.DataFrame, out: Path):
    """
    Bar chart of final top-1% recall, grouped by backbone.
    Best UQ+acq combo per backbone shown, with error bars across combos.
    """
    final = df[df["round"] == df["round"].max()].copy()
    summary = (final
               .groupby("backbone")["top1pct_recall"]
               .agg(["mean", "std", "max"])
               .reset_index())

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary))

    bars = ax.bar(x, summary["mean"] * 100,
                  yerr=summary["std"] * 100,
                  color=[COLORS[b] for b in summary["backbone"]],
                  capsize=5, width=0.5, alpha=0.85)

    # annotate max
    for xi, (_, row) in zip(x, summary.iterrows()):
        ax.text(xi, row["max"] * 100 + 1,
                f"max {row['max']*100:.1f}%",
                ha="center", fontsize=8, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(summary["backbone"], fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Top-1% recall (final round)")
    ax.set_title("Final recall by backbone\n(mean ± std across UQ × acq combos)",
                 fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] {out}")


def plot_heatmap(df: pd.DataFrame, out: Path):
    """Heatmap: backbone × (uq+acq) → final top-1% recall."""
    final  = df[df["round"] == df["round"].max()].copy()
    final["combo"] = final["uq"] + "\n" + final["acq"]
    pivot  = final.pivot_table(
        index="backbone", columns="combo",
        values="top1pct_recall", aggfunc="mean"
    ) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto",
                   vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8,
                        color="black" if val < 60 else "white")

    plt.colorbar(im, ax=ax, label="Top-1% recall (%)")
    ax.set_title("Final top-1% recall: backbone × (UQ / acquisition)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] {out}")


def print_summary_table(df: pd.DataFrame):
    final = df[df["round"] == df["round"].max()].copy()
    final = final.sort_values("top1pct_recall", ascending=False)
    print("\n" + "="*70)
    print(f"{'Backbone':<12} {'UQ':<12} {'Acq':<10} "
          f"{'Recall (top-1%)':<18} {'Best score'}")
    print("="*70)
    for _, row in final.iterrows():
        print(f"{row['backbone']:<12} {row['uq']:<12} {row['acq']:<10} "
              f"{row['top1pct_recall']*100:>10.1f}%          "
              f"{row['best_score']:.3f} kcal/mol")


# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="Enamine50k")
    p.add_argument("--n-rounds",   type=int, default=15)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--init-size",  type=int, default=200)
    p.add_argument("--epochs",     type=int, default=80)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--plot-only",  action="store_true",
                   help="Skip training, just load existing runs and plot")
    p.add_argument("--force",      action="store_true",
                   help="Re-run even if results already exist")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.plot_only:
        run_all(args)

    df      = load_results(args.dataset)
    out_dir = ROOT / "results" / "comparison" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_top1pct_recall(df, out_dir / "recall_vs_round.png")
    plot_best_score    (df, out_dir / "best_score_vs_calls.png")
    plot_final_bar     (df, out_dir / "final_recall_bar.png")
    plot_heatmap       (df, out_dir / "heatmap.png")
    print_summary_table(df)