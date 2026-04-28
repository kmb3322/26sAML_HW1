"""Training entry point for the sanity check and modular arithmetic tasks.

Usage:
    python train.py --config configs/sanity.yaml
    python train.py --config configs/add_p97_l1_seed0.yaml
    python train.py --task modular --op + --p 97 --n_layer 1 --seed 0 --out_dir runs/foo

CLI flags override YAML values, which override the dataclass defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from model import GPT, GPTConfig
from tokenizer_data import (
    ResidueTokenizer,
    SanityTokenizer,
    make_modular_rows,
    make_sanity_examples,
    rows_to_examples,
)


# ---------------------------------------------------------------------------
# Batch collation + masked cross-entropy
# (formerly in train_utils.py; inlined to keep file count minimal)
# ---------------------------------------------------------------------------


def collate_lm_batch(
    batch: List[Dict],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences for next-token language model training.

    Returns:
      x: ``[B, T-1]`` token ids (pad-filled)
      y: ``[B, T-1]`` next-token targets (-100 for pad, ignored by CE)
      target_mask: ``[B, T-1]`` 1.0 at positions where the loss should count
    """
    max_len = max(len(ex["ids"]) for ex in batch)
    B = len(batch)
    T = max_len - 1

    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), -100, dtype=torch.long)
    target_mask = torch.zeros((B, T), dtype=torch.float32)

    for i, ex in enumerate(batch):
        ids = torch.tensor(ex["ids"], dtype=torch.long)
        m = torch.tensor(ex["target_mask"], dtype=torch.float32)
        seq_T = len(ids) - 1
        x[i, :seq_T] = ids[:-1]
        y[i, :seq_T] = ids[1:]
        target_mask[i, :seq_T] = m

    return x, y, target_mask


def masked_cross_entropy(
    logits: torch.Tensor,
    y: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean cross-entropy across positions where ``target_mask`` is non-zero."""
    B, T, V = logits.shape
    per_token_loss = F.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T),
        reduction="none",
        ignore_index=-100,
    ).reshape(B, T)
    denom = target_mask.sum().clamp_min(1.0)
    return (per_token_loss * target_mask).sum() / denom


# ---------------------------------------------------------------------------
# Per-run plotting (formerly in plot_metrics.py)
# ---------------------------------------------------------------------------


def plot_one_run(csv_path: str, out_dir: str, title: str = "") -> None:
    """Read a metrics.csv and write loss.png + acc.png next to it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

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


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    # Task
    task: str = "modular"            # "modular" | "sanity"
    op: str = "+"                    # for modular
    p: int = 97                      # for modular
    sanity_prompt_len: int = 0       # for sanity: how many leading tokens are prompt

    # Data
    seed: int = 0
    train_frac: float = 0.50
    val_frac: float = 0.10
    max_p: int = 113                 # tokenizer range for modular task

    # Model
    n_layer: int = 1
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 16
    dropout: float = 0.0
    bias: bool = True

    # Optimizer
    optimizer: str = "adam"          # "adam" | "adamw"
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Training
    batch_size: int = 512
    max_steps: int = 100_000
    eval_every: int = 500
    eval_batch_size: int = 2048

    # IO
    out_dir: str = "runs/debug"
    run_name: Optional[str] = None
    auto_plot: bool = True           # write loss.png/acc.png next to metrics.csv at end


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML at {path} must be a mapping, got {type(data)}.")
    return data


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_loss(
    model: GPT,
    examples: List[Dict],
    pad_id: int,
    device: torch.device,
    batch_size: int = 2048,
) -> float:
    if not examples:
        return float("nan")

    was_training = model.training
    model.eval()

    total_loss_sum = 0.0
    total_mask = 0.0

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        x, y, mask = collate_lm_batch(batch, pad_id=pad_id)
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        logits = model(x)
        per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(y.shape)

        total_loss_sum += float((per_token * mask).sum().item())
        total_mask += float(mask.sum().item())

    if was_training:
        model.train()
    return total_loss_sum / max(total_mask, 1.0)


@torch.no_grad()
def modular_answer_accuracy(
    model: GPT,
    rows: List[Dict],
    tokenizer: ResidueTokenizer,
    p: int,
    device: torch.device,
    batch_size: int = 2048,
) -> float:
    if not rows:
        return float("nan")

    was_training = model.training
    model.eval()

    candidate_ids = torch.tensor([tokenizer.num_id(i) for i in range(p)], device=device)

    correct = 0
    total = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [tokenizer.encode_problem(r["a"], r["b"], r["op"], c=None) for r in chunk]
        x = torch.tensor(prompts, dtype=torch.long, device=device)

        logits = model(x)[:, -1, :]
        restricted = logits.index_select(1, candidate_ids)
        pred_offsets = restricted.argmax(dim=-1)
        pred_token_ids = candidate_ids[pred_offsets].detach().cpu().tolist()
        preds = [tokenizer.id_to_num(tid) for tid in pred_token_ids]
        labels = [r["c"] for r in chunk]

        correct += sum(int(a == b) for a, b in zip(preds, labels))
        total += len(chunk)

    if was_training:
        model.train()
    return correct / max(total, 1)


@torch.no_grad()
def sanity_full_match(
    model: GPT,
    tokenizer: SanityTokenizer,
    device: torch.device,
    prompt_len: int = 0,
    max_new_tokens: int = 16,
) -> bool:
    """Greedy-generate the suffix from a ``prompt_len``-token prompt.

    With ``prompt_len <= 1`` the prompt is just ``[BOS]`` and we expect the
    full sentence ``[I, love, machine, learning, EOS]`` back.  With
    ``prompt_len = 3`` the prompt is ``[BOS, I, love]`` and we expect
    ``[machine, learning, EOS]``.
    """
    full = tokenizer.encode_words(list(tokenizer.words))  # [BOS, I, love, machine, learning, EOS]
    cut = max(prompt_len, 1)
    prompt_ids = full[:cut]
    expected_ids = full[cut:]

    was_training = model.training
    model.eval()

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        next_id = int(logits.argmax(dim=-1).item())
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == tokenizer.eos_id:
            break

    if was_training:
        model.train()
    return generated == expected_ids


# ---------------------------------------------------------------------------
# Optimizer + checkpointing
# ---------------------------------------------------------------------------


def build_optimizer(model: GPT, cfg: RunConfig):
    betas = (cfg.beta1, cfg.beta2)
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=betas,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=betas,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def save_checkpoint(
    out_dir: Path,
    model: GPT,
    tokenizer,
    cfg: RunConfig,
    step: int,
    metrics: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "tokenizer": tokenizer.to_dict(),
        "run_config": asdict(cfg),
        "metrics": metrics,
    }
    torch.save(ckpt, out_dir / "ckpt.pt")
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "tokenizer.json", "w") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)


def sample_batch(examples: List[Dict], batch_size: int, pad_id: int, rng: random.Random):
    batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
    return collate_lm_batch(batch, pad_id=pad_id)


def write_metrics_row(log_path: Path, row: Dict, fieldnames: List[str]) -> None:
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: RunConfig) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- task-specific data + tokenizer ----
    if cfg.task == "modular":
        tokenizer = ResidueTokenizer(max_p=cfg.max_p)
        splits = make_modular_rows(
            op=cfg.op,
            p=cfg.p,
            seed=cfg.seed,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
        )
        train_examples = rows_to_examples(splits["train"], tokenizer)
        val_examples = rows_to_examples(splits["val"], tokenizer)
        test_examples = rows_to_examples(splits["test"], tokenizer)
        print(
            f"Modular task: op={cfg.op} p={cfg.p} | "
            f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
        )

    elif cfg.task == "sanity":
        tokenizer = SanityTokenizer()
        train_examples = make_sanity_examples(
            tokenizer,
            prompt_len=cfg.sanity_prompt_len,
        )
        val_examples = []
        test_examples = []
        splits = {"train": [], "val": [], "test": []}
        print(f"Sanity task | prompt_len={cfg.sanity_prompt_len}")

    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    # ---- model + optimizer ----
    gpt_cfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=tokenizer.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = GPT(gpt_cfg).to(device)
    optimizer = build_optimizer(model, cfg)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ---- logging ----
    log_path = out_dir / "metrics.csv"
    if log_path.exists():
        log_path.unlink()
    fieldnames = [
        "step",
        "train_loss", "val_loss", "test_loss",
        "train_acc", "val_acc", "test_acc",
    ]

    rng = random.Random(cfg.seed + 1)
    best_tracker_acc = -1.0
    latest_metrics: Dict[str, float] = {}

    pbar = trange(cfg.max_steps + 1)
    for step in pbar:
        # ---- evaluation ----
        if step % cfg.eval_every == 0:
            metrics: Dict[str, float] = {"step": step}

            if cfg.task == "modular":
                metrics["train_loss"] = eval_loss(
                    model, train_examples, tokenizer.pad_id, device, cfg.eval_batch_size
                )
                metrics["val_loss"] = eval_loss(
                    model, val_examples, tokenizer.pad_id, device, cfg.eval_batch_size
                )
                metrics["test_loss"] = eval_loss(
                    model, test_examples, tokenizer.pad_id, device, cfg.eval_batch_size
                )
                metrics["train_acc"] = modular_answer_accuracy(
                    model, splits["train"], tokenizer, cfg.p, device, cfg.eval_batch_size
                )
                metrics["val_acc"] = modular_answer_accuracy(
                    model, splits["val"], tokenizer, cfg.p, device, cfg.eval_batch_size
                )
                metrics["test_acc"] = modular_answer_accuracy(
                    model, splits["test"], tokenizer, cfg.p, device, cfg.eval_batch_size
                )
                # Best-checkpoint selection uses val only (no test leakage).
                tracker_acc = metrics["val_acc"]

            else:  # sanity
                metrics["train_loss"] = eval_loss(
                    model, train_examples, tokenizer.pad_id, device, cfg.eval_batch_size
                )
                metrics["val_loss"] = float("nan")
                metrics["test_loss"] = float("nan")
                match = sanity_full_match(model, tokenizer, device, prompt_len=cfg.sanity_prompt_len)
                metrics["train_acc"] = float(match)
                metrics["val_acc"] = float("nan")
                metrics["test_acc"] = float("nan")
                tracker_acc = metrics["train_acc"]

            latest_metrics = metrics
            write_metrics_row(log_path, metrics, fieldnames)

            postfix = {
                "loss": f"{metrics['train_loss']:.4f}" if metrics["train_loss"] == metrics["train_loss"] else "nan",
            }
            if metrics["val_acc"] == metrics["val_acc"]:
                postfix["val_acc"] = f"{metrics['val_acc']:.3f}"
                postfix["test_acc"] = f"{metrics['test_acc']:.3f}"
            else:
                postfix["match"] = f"{metrics['train_acc']:.0f}"
            pbar.set_postfix(postfix)

            if tracker_acc == tracker_acc and tracker_acc > best_tracker_acc:
                best_tracker_acc = tracker_acc
                save_checkpoint(out_dir / "best", model, tokenizer, cfg, step, metrics)
            save_checkpoint(out_dir / "latest", model, tokenizer, cfg, step, metrics)

        if step == cfg.max_steps:
            break

        # ---- one training step ----
        x, y, mask = sample_batch(train_examples, cfg.batch_size, tokenizer.pad_id, rng)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = model(x)
        loss = masked_cross_entropy(logits, y, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    save_checkpoint(out_dir / "final", model, tokenizer, cfg, cfg.max_steps, latest_metrics)
    print(f"Done. Final metrics: {latest_metrics}")
    print(f"Saved checkpoints under: {out_dir}")

    if cfg.auto_plot:
        try:
            run_label = cfg.run_name or out_dir.name
            plot_one_run(str(log_path), str(out_dir), title=f"{run_label} ({cfg.task})")
            print(f"Saved per-run plots to: {out_dir}/loss.png, {out_dir}/acc.png")
        except Exception as e:
            print(f"[auto_plot] skipped: {e!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes", "y"):
        return True
    if s.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {s!r}.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small transformer (sanity / modular).")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")

    parser.add_argument("--task", type=str, default=None, choices=["modular", "sanity"])
    parser.add_argument("--op", type=str, default=None, choices=["+", "-", "/"])
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--sanity_prompt_len", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_frac", type=float, default=None)
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--max_p", type=int, default=None)

    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--bias", type=_parse_bool, default=None)

    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--auto_plot", type=_parse_bool, default=None)

    return parser


def parse_cli_to_cfg(argv: Optional[List[str]] = None) -> RunConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = RunConfig()
    valid = {f.name for f in fields(RunConfig)}

    if args.config is not None:
        yaml_data = load_yaml_config(args.config)
        unknown = set(yaml_data) - valid
        if unknown:
            print(f"Warning: ignoring unknown YAML keys: {sorted(unknown)}")
        cfg = RunConfig(**{**asdict(cfg), **{k: v for k, v in yaml_data.items() if k in valid}})

    overrides: Dict[str, Any] = {}
    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        if k in valid:
            overrides[k] = v
    cfg = RunConfig(**{**asdict(cfg), **overrides})
    return cfg


def main():
    cfg = parse_cli_to_cfg()
    print("Run config:")
    print(json.dumps(asdict(cfg), indent=2))
    train(cfg)


if __name__ == "__main__":
    main()
