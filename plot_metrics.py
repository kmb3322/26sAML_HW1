"""Plotting utilities for ``metrics.csv`` produced by ``train.py``.

Examples:
    # Single-run plot:
    python plot_metrics.py --csv runs/add_p97_l1_seed0/metrics.csv \
        --out_dir plots/add_p97_l1_seed0 --title "p=97, +, 1-layer"

    # Random-restart overlay:
    python plot_metrics.py --csvs \
        runs/add_p97_l1_seed0/metrics.csv \
        runs/add_p97_l1_seed1/metrics.csv \
        runs/add_p97_l1_seed2/metrics.csv \
        --labels seed0 seed1 seed2 \
        --out_dir plots/add_p97_l1_restarts \
        --title "Random restarts: p=97, +, 1-layer"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_one_run(csv_path: str, out_dir: str, title: str = "") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    plt.figure()
    for col, lbl in [("train_loss", "train"), ("val_loss", "val"), ("test_loss", "test")]:
        if col in df and df[col].notna().any():
            plt.plot(df["step"], df[col], label=lbl)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title(title or "loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loss.png", dpi=200)
    plt.close()

    plt.figure()
    for col, lbl in [("train_acc", "train"), ("val_acc", "val"), ("test_acc", "test")]:
        if col in df and df[col].notna().any():
            plt.plot(df["step"], df[col], label=lbl)
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title(title or "accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "acc.png", dpi=200)
    plt.close()


def plot_seeds_overlay(
    csv_paths: List[str],
    labels: List[str],
    out_dir: str,
    title: str = "random restarts",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for path, lbl in zip(csv_paths, labels):
        df = pd.read_csv(path)
        plt.plot(df["step"], df["train_acc"], alpha=0.7, label=f"{lbl} train")
        plt.plot(df["step"], df["test_acc"], alpha=0.7, linestyle="--", label=f"{lbl} test")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "seeds_acc.png", dpi=200)
    plt.close()


def plot_grokking_summary(
    csv_path: str,
    out_dir: str,
    title: str = "grokking",
    threshold: float = 0.99,
) -> Optional[dict]:
    """Plot loss/acc with markers for the train- and test-solved steps.

    Returns ``{"train_solved_step", "test_solved_step", "delay"}`` or ``None``
    if neither curve crosses ``threshold``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    def first_crossing(col: str) -> Optional[int]:
        crossed = df[df[col] >= threshold]
        if crossed.empty:
            return None
        return int(crossed["step"].iloc[0])

    train_step = first_crossing("train_acc")
    test_step = first_crossing("test_acc")

    plt.figure()
    plt.plot(df["step"], df["train_acc"], label="train acc")
    plt.plot(df["step"], df["test_acc"], label="test acc")
    if train_step is not None:
        plt.axvline(train_step, color="C0", linestyle=":", alpha=0.6, label=f"train>={threshold} @ {train_step}")
    if test_step is not None:
        plt.axvline(test_step, color="C1", linestyle=":", alpha=0.6, label=f"test>={threshold} @ {test_step}")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "grokking_acc.png", dpi=200)
    plt.close()

    summary = {
        "train_solved_step": train_step,
        "test_solved_step": test_step,
        "delay": (test_step - train_step) if (train_step is not None and test_step is not None) else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Single metrics.csv path.")
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--title", default="")
    parser.add_argument("--csvs", nargs="*", default=None, help="Multiple csv paths for overlay.")
    parser.add_argument("--labels", nargs="*", default=None, help="Labels paired with --csvs.")
    parser.add_argument("--grokking", action="store_true", help="Also annotate grokking thresholds for --csv.")
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()

    if args.csv:
        plot_one_run(args.csv, args.out_dir, title=args.title)
        if args.grokking:
            summary = plot_grokking_summary(
                args.csv, args.out_dir, title=args.title or "grokking", threshold=args.threshold
            )
            print("grokking summary:", summary)

    if args.csvs:
        labels = args.labels or [Path(c).parent.name for c in args.csvs]
        plot_seeds_overlay(args.csvs, labels, args.out_dir, title=args.title or "random restarts")


if __name__ == "__main__":
    main()
