"""Aggregate Part 1 results into report-ready figures + summary CSVs.

Reads ``runs/<name>/metrics.csv`` and writes everything under ``plots/``.

Generated artefacts:

  plots/addsub/<run>/{loss.png, acc.png}                  (per-run, also
      already produced by train.py auto_plot, copied here for one-stop access)
  plots/addsub/p97_add_l1_restarts/seeds_acc.png          (3-seed overlay)
  plots/grokking/div_p97/grokking_acc.png                 (annotated)
  plots/grokking/div_p97/loss.png, acc.png
  plots/ablation/wd/comparison_acc.png                    (4 wd's overlaid)
  plots/ablation/tf/comparison_acc.png                    (5 tf's overlaid)
  plots/ablation/summary.csv                              (delay table)

Usage:
    python make_part1_plots.py
    python make_part1_plots.py --runs_dir runs --plots_dir plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from plot_metrics import plot_one_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def first_crossing(df: pd.DataFrame, col: str, threshold: float) -> Optional[int]:
    """First step at which ``df[col] >= threshold``, or None."""
    if col not in df:
        return None
    crossed = df[df[col] >= threshold]
    if crossed.empty:
        return None
    return int(crossed["step"].iloc[0])


def safe_load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def per_run_plot(run_dir: Path, plot_dir: Path, title: str) -> None:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        print(f"[skip] {csv_path} not found")
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_one_run(str(csv_path), str(plot_dir), title=title)


# ---------------------------------------------------------------------------
# Section 1.2: addition/subtraction
# ---------------------------------------------------------------------------


ADDSUB_RUNS: List[Tuple[str, str]] = [
    ("add_p97_l1_seed0",  "+ p=97 1L"),
    ("add_p97_l2_seed0",  "+ p=97 2L"),
    ("add_p113_l1_seed0", "+ p=113 1L"),
    ("add_p113_l2_seed0", "+ p=113 2L"),
    ("sub_p97_l1_seed0",  "- p=97 1L"),
    ("sub_p97_l2_seed0",  "- p=97 2L"),
    ("sub_p113_l1_seed0", "- p=113 1L"),
    ("sub_p113_l2_seed0", "- p=113 2L"),
]


def make_addsub_plots(runs_dir: Path, plots_dir: Path) -> None:
    out_root = plots_dir / "addsub"
    out_root.mkdir(parents=True, exist_ok=True)

    for run_name, label in ADDSUB_RUNS:
        per_run_plot(runs_dir / run_name, out_root / run_name, title=label)

    # Random-restart overlay for (p=97, +, 1-layer)
    seeds = []
    for seed in (0, 1, 2):
        sub = runs_dir / f"add_p97_l1_seed{seed}"
        csv = sub / "metrics.csv"
        if csv.exists():
            seeds.append((seed, pd.read_csv(csv)))

    if len(seeds) >= 2:
        plt.figure()
        for seed, df in seeds:
            plt.plot(df["step"], df["train_acc"], alpha=0.7, label=f"seed{seed} train")
            plt.plot(df["step"], df["test_acc"], alpha=0.7, linestyle="--", label=f"seed{seed} test")
        plt.xlabel("step")
        plt.ylabel("accuracy")
        plt.ylim(-0.02, 1.02)
        plt.title("Random restarts: + p=97 1-layer")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out = out_root / "p97_add_l1_restarts"
        out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "seeds_acc.png", dpi=200)
        plt.close()

        plt.figure()
        for seed, df in seeds:
            plt.plot(df["step"], df["train_loss"], alpha=0.7, label=f"seed{seed} train")
            plt.plot(df["step"], df["test_loss"], alpha=0.7, linestyle="--", label=f"seed{seed} test")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.title("Random restarts: + p=97 1-layer")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "seeds_loss.png", dpi=200)
        plt.close()
    else:
        print("[skip] random-restart overlay (need >=2 seeds in runs/add_p97_l1_seed{0,1,2})")


# ---------------------------------------------------------------------------
# Section 1.3: grokking
# ---------------------------------------------------------------------------


GROKKING_RUN = "div_p97_grokking_seed0"


def make_grokking_plot(runs_dir: Path, plots_dir: Path, threshold: float = 0.99) -> Optional[Dict]:
    run_dir = runs_dir / GROKKING_RUN
    csv = run_dir / "metrics.csv"
    if not csv.exists():
        print(f"[skip] grokking: {csv} not found")
        return None

    out = plots_dir / "grokking" / GROKKING_RUN
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)

    # Standard loss/acc plots
    plot_one_run(str(csv), str(out), title="Grokking: / p=97 (AdamW, wd=1.0)")

    # Annotated grokking acc plot
    train_step = first_crossing(df, "train_acc", threshold)
    test_step = first_crossing(df, "test_acc", threshold)

    plt.figure()
    plt.plot(df["step"], df["train_acc"], label="train acc")
    plt.plot(df["step"], df["test_acc"], label="test acc")
    if train_step is not None:
        plt.axvline(train_step, color="C0", linestyle=":", alpha=0.6,
                    label=f"train>={threshold} @ {train_step}")
    if test_step is not None:
        plt.axvline(test_step, color="C1", linestyle=":", alpha=0.6,
                    label=f"test>={threshold} @ {test_step}")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.ylim(-0.02, 1.02)
    plt.xscale("log")
    plt.title("Grokking on division (p=97), 2-layer, AdamW wd=1.0")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "grokking_acc.png", dpi=200)
    plt.close()

    return {
        "run": GROKKING_RUN,
        "train_solved_step": train_step,
        "test_solved_step": test_step,
        "delay": (test_step - train_step) if (train_step is not None and test_step is not None) else None,
    }


# ---------------------------------------------------------------------------
# Section 1.4: ablations
# ---------------------------------------------------------------------------


# Each entry: (run_name, value, value_label)
ABLATION_WD: List[Tuple[str, float, str]] = [
    ("abl_wd_0.0", 0.0, "wd=0.0"),
    ("abl_wd_0.1", 0.1, "wd=0.1"),
    ("abl_wd_0.5", 0.5, "wd=0.5"),
    (GROKKING_RUN, 1.0, "wd=1.0 (baseline)"),
]
ABLATION_TF: List[Tuple[str, float, str]] = [
    ("abl_tf_0.30", 0.30, "tf=0.30"),
    ("abl_tf_0.40", 0.40, "tf=0.40"),
    (GROKKING_RUN, 0.50, "tf=0.50 (baseline)"),
    ("abl_tf_0.60", 0.60, "tf=0.60"),
    ("abl_tf_0.80", 0.80, "tf=0.80"),
]


def _plot_ablation(
    runs_dir: Path,
    out_dir: Path,
    runs: List[Tuple[str, float, str]],
    title: str,
) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict] = []

    plt.figure(figsize=(8, 5))
    for run_name, value, label in runs:
        df = safe_load_csv(runs_dir / run_name / "metrics.csv")
        if df is None:
            print(f"[skip] {run_name}: metrics.csv missing")
            continue
        plt.plot(df["step"], df["train_acc"], alpha=0.4, linestyle="--",
                 label=f"{label} train")
        plt.plot(df["step"], df["test_acc"], alpha=0.9, label=f"{label} test")

        train_step = first_crossing(df, "train_acc", 0.99)
        test_step = first_crossing(df, "test_acc", 0.99)
        delay = (test_step - train_step) if (train_step is not None and test_step is not None) else None
        summary.append({
            "run": run_name,
            "value": value,
            "label": label,
            "train_solved_step": train_step,
            "test_solved_step": test_step,
            "delay": delay,
        })

    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.ylim(-0.02, 1.02)
    plt.xscale("log")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_acc.png", dpi=200)
    plt.close()

    return summary


def make_ablation_plots(runs_dir: Path, plots_dir: Path) -> List[Dict]:
    out_root = plots_dir / "ablation"
    summary: List[Dict] = []

    s_wd = _plot_ablation(
        runs_dir,
        out_root / "wd",
        ABLATION_WD,
        title="Ablation A: weight decay (division p=97, 2-layer)",
    )
    for r in s_wd:
        r["ablation"] = "weight_decay"
        summary.append(r)

    s_tf = _plot_ablation(
        runs_dir,
        out_root / "tf",
        ABLATION_TF,
        title="Ablation B: train fraction (division p=97, 2-layer)",
    )
    for r in s_tf:
        r["ablation"] = "train_frac"
        summary.append(r)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_summary_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ablation", "run", "label", "value",
                  "train_solved_step", "test_solved_step", "delay"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            # Normalize possibly-missing keys.
            w.writerow({k: row.get(k) for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--threshold", type=float, default=0.99,
                        help="Accuracy threshold for grokking-delay measurement.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=== Section 1.2 addition/subtraction ===")
    make_addsub_plots(runs_dir, plots_dir)

    print("=== Section 1.3 grokking ===")
    grokking = make_grokking_plot(runs_dir, plots_dir, threshold=args.threshold)
    if grokking is not None:
        print(f"  grokking summary: {grokking}")

    print("=== Section 1.4 ablations ===")
    abl_summary = make_ablation_plots(runs_dir, plots_dir)
    if grokking is not None:
        # Add grokking baseline to the summary too for one-stop comparison.
        abl_summary.append({
            "ablation": "baseline",
            "run": grokking["run"],
            "label": "grokking baseline",
            "value": None,
            "train_solved_step": grokking["train_solved_step"],
            "test_solved_step": grokking["test_solved_step"],
            "delay": grokking["delay"],
        })

    summary_path = plots_dir / "ablation" / "summary.csv"
    write_summary_csv(abl_summary, summary_path)
    print(f"  wrote {summary_path}")

    print("\nAll plots written under:", plots_dir.resolve())


if __name__ == "__main__":
    main()
