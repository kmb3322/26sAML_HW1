# Advanced ML HW 1

CSE 493S HW 1 (Spring 2026) — training infrastructure for the sanity check
and modular-arithmetic experiments (Parts 0 and 1).  See `ps.md` / `ps.pdf`
for the original problem set.

---

## Layout

```
.
├── model.py               # nano-GPT (starter; unchanged)
├── tokenizer_data.py      # residue & sanity tokenizers + data generation
├── train.py               # training loop + collate + masked CE + per-run plots
├── inference.py           # checkpoint loading, prediction, generation
├── part_0_1_contract.py   # autograder interface
├── configs/               # YAML config per experiment
├── runs/                  # checkpoints + metrics.csv (git-ignored)
└── plots/                 # report-ready cross-run figures (git-ignored)
```

In this assignment every batch contains fixed-length sequences (modular
task is always 7 tokens, sanity is a single sequence of length 6), so no
pad-attention masking is needed and `model.py` is used as-is.

---

## Setup

```bash
pip install -r requirements.txt
```

Tested on PyTorch 2.x with both CPU and CUDA.

---

## Configs at a glance

Every YAML file in `configs/` is a 1:1 mapping to the `RunConfig` dataclass
in `train.py`; running `python train.py --config configs/X.yaml` is just
shorthand for setting all those fields via CLI.  CLI overrides (`--seed 1`,
`--out_dir runs/foo`, ...) take precedence.

```
configs/
  sanity.yaml                 # Part 0:    full-token loss
  sanity_masked.yaml          # Part 0:    prompt_len=3, train on suffix
  smoke_modular.yaml          # 1k-step smoke for the modular pipeline

  add_p{97,113}_l{1,2}_seed0.yaml,  sub_p{97,113}_l{1,2}_seed0.yaml
                              # Part 1.2: 8 main warmup runs

  div_p97_grokking_seed0.yaml # Part 1.3: AdamW, wd=1.0, 100k steps

  abl_wd_{0.0,0.1,0.5}.yaml   # Part 1.4 ablation A: weight decay
  abl_tf_{0.30,0.40,0.60,0.80}.yaml
                              # Part 1.4 ablation B: train fraction
```

The `(p=97, +, 1-layer)` random-restart triple uses CLI overrides on
`add_p97_l1_seed0.yaml`: `--seed {0,1,2} --out_dir runs/add_p97_l1_seed{0,1,2}`.

---

## Running experiments

### Single run

```bash
python train.py --config configs/sanity.yaml
python train.py --config configs/add_p97_l1_seed0.yaml
python train.py --config configs/div_p97_grokking_seed0.yaml
```

Each invocation writes:

```
runs/<name>/
    metrics.csv
    loss.png  acc.png        # per-run, auto-generated at end of training
    best/   latest/   final/
        ckpt.pt   config.json   tokenizer.json
```

Smoke test before any long run:

```bash
python train.py --config configs/smoke_modular.yaml   # ~1 min on CPU
```

### Batch (Part 1)

A small shell loop runs everything; already-finished runs are skipped via
the `--skip_if_done` convention by checking `final/ckpt.pt`.

```bash
# Part 0 sanity
for cfg in configs/sanity.yaml configs/sanity_masked.yaml; do
    python train.py --config "$cfg"
done

# Part 1.2: addition / subtraction (8 runs)
for cfg in configs/add_p97_l1_seed0.yaml configs/add_p97_l2_seed0.yaml \
           configs/add_p113_l1_seed0.yaml configs/add_p113_l2_seed0.yaml \
           configs/sub_p97_l1_seed0.yaml configs/sub_p97_l2_seed0.yaml \
           configs/sub_p113_l1_seed0.yaml configs/sub_p113_l2_seed0.yaml; do
    python train.py --config "$cfg"
done

# Random restarts for (p=97, +, 1-layer)
for s in 1 2; do
    python train.py --config configs/add_p97_l1_seed0.yaml \
        --seed $s --out_dir runs/add_p97_l1_seed${s}
done

# Part 1.3: grokking
python train.py --config configs/div_p97_grokking_seed0.yaml

# Part 1.4: ablations
for cfg in configs/abl_wd_0.0.yaml configs/abl_wd_0.1.yaml configs/abl_wd_0.5.yaml \
           configs/abl_tf_0.30.yaml configs/abl_tf_0.40.yaml \
           configs/abl_tf_0.60.yaml configs/abl_tf_0.80.yaml; do
    python train.py --config "$cfg"
done
```

### Inference

```bash
# Modular addition
python inference.py --checkpoint_dir runs/add_p97_l1_seed0/best \
    --a 12 --b 34 --op + --p 97

# Sanity-check generation (full sentence from BOS)
python inference.py --checkpoint_dir runs/sanity/final

# Sanity-check generation with a partial prompt (BOS prepended automatically)
python inference.py --checkpoint_dir runs/sanity_masked/final --sanity_prompt "I love"
```

---

## Cross-run plots (paste into a Colab/Python cell)

Each `train.py` run already writes `loss.png`/`acc.png` next to its own
`metrics.csv`.  The four report-ready cross-run figures below are short
pandas+matplotlib snippets that read those CSVs.

### Random-restart overlay (Part 1.2)

```python
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

Path("plots/restarts").mkdir(parents=True, exist_ok=True)
plt.figure()
for s in (0, 1, 2):
    df = pd.read_csv(f"runs/add_p97_l1_seed{s}/metrics.csv")
    plt.plot(df["step"], df["train_acc"], alpha=0.7, label=f"seed{s} train")
    plt.plot(df["step"], df["test_acc"], alpha=0.7, ls="--", label=f"seed{s} test")
plt.xlabel("step"); plt.ylabel("accuracy"); plt.ylim(-0.02, 1.02)
plt.title("Random restarts: + p=97 1-layer"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("plots/restarts/seeds_acc.png", dpi=200)
```

### Grokking annotated (Part 1.3)

```python
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

THR = 0.99
df = pd.read_csv("runs/div_p97_grokking_seed0/metrics.csv")
def first(col):
    c = df[df[col] >= THR]
    return None if c.empty else int(c["step"].iloc[0])
ts, vs = first("train_acc"), first("test_acc")

Path("plots/grokking").mkdir(parents=True, exist_ok=True)
plt.figure()
plt.plot(df["step"], df["train_acc"], label="train")
plt.plot(df["step"], df["test_acc"],  label="test")
if ts: plt.axvline(ts, ls=":", color="C0", alpha=0.6, label=f"train≥{THR} @ {ts}")
if vs: plt.axvline(vs, ls=":", color="C1", alpha=0.6, label=f"test≥{THR}  @ {vs}")
plt.xscale("log"); plt.ylim(-0.02, 1.02)
plt.xlabel("step"); plt.ylabel("accuracy")
plt.title("Grokking: / p=97 (AdamW, wd=1.0)"); plt.legend(fontsize=9)
plt.tight_layout(); plt.savefig("plots/grokking/grokking_acc.png", dpi=200)
print(f"train_solved={ts}  test_solved={vs}  delay={None if vs is None or ts is None else vs-ts}")
```

### Ablation comparison + summary table (Part 1.4)

```python
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

ABLATIONS = {
    "weight_decay": [
        ("abl_wd_0.0", "wd=0.0"),
        ("abl_wd_0.1", "wd=0.1"),
        ("abl_wd_0.5", "wd=0.5"),
        ("div_p97_grokking_seed0", "wd=1.0 (baseline)"),
    ],
    "train_frac": [
        ("abl_tf_0.30", "tf=0.30"),
        ("abl_tf_0.40", "tf=0.40"),
        ("div_p97_grokking_seed0", "tf=0.50 (baseline)"),
        ("abl_tf_0.60", "tf=0.60"),
        ("abl_tf_0.80", "tf=0.80"),
    ],
}
THR = 0.99
def first(df, col):
    c = df[df[col] >= THR]
    return None if c.empty else int(c["step"].iloc[0])

rows = []
for name, runs in ABLATIONS.items():
    Path(f"plots/ablation/{name}").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for run, label in runs:
        csv = Path(f"runs/{run}/metrics.csv")
        if not csv.exists():
            print("skip", csv); continue
        df = pd.read_csv(csv)
        plt.plot(df["step"], df["train_acc"], alpha=0.4, ls="--", label=f"{label} train")
        plt.plot(df["step"], df["test_acc"],  alpha=0.9,         label=f"{label} test")
        ts, vs = first(df, "train_acc"), first(df, "test_acc")
        rows.append({
            "ablation": name, "run": run, "label": label,
            "train_solved_step": ts, "test_solved_step": vs,
            "delay": (vs - ts) if (ts is not None and vs is not None) else None,
        })
    plt.xscale("log"); plt.ylim(-0.02, 1.02)
    plt.xlabel("step"); plt.ylabel("accuracy")
    plt.title(f"Ablation: {name} (division p=97, 2-layer)")
    plt.legend(fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(f"plots/ablation/{name}/comparison_acc.png", dpi=200)
    plt.close()

pd.DataFrame(rows).to_csv("plots/ablation/summary.csv", index=False)
print(pd.DataFrame(rows))
```

---

## Training infrastructure design

- **Tokenizer.** `ResidueTokenizer` maps each integer `0..max_p-1` to a
  single token, so every modular problem is exactly 7 tokens
  (`[BOS, a, op, b, =, c, EOS]`).  `max_p=113` covers both `p=97` and
  `p=113` experiments.
- **Loss masking.** Next-token targets live in `ids[1:]`.  The answer `c`
  is at target index `len(prompt)−1` (right after `=`); only that index is
  unmasked.  For sanity #2, `prompt_len=3` masks the first two target
  positions so the trained targets are exactly `[machine, learning, EOS]`.
- **Splits.** For `+`/`-` the dataset is `p²`.  For `/` we exclude `b=0`
  (no modular inverse), giving `p(p−1)`.  Default 50% train / 10% val /
  40% test, shuffled per-run seed.
- **Best checkpoint.** Tracks `val_acc` (modular) / full-string greedy
  match (sanity); test set is logged for plotting only.
- **Config.** `train.py` accepts `--config <yaml>` plus arbitrary CLI
  overrides (CLI > YAML > dataclass defaults).  Auto-plotting is on by
  default; disable with `--auto_plot false`.

---

## Local ↔ Colab workflow

```python
!git clone https://github.com/kmb3322/AML_HW1.git
%cd AML_HW1
!pip install -q -r requirements.txt

# Persist runs/ to Drive so checkpoints survive runtime restarts.
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/cse493s_hw1_runs
!ln -sfn /content/drive/MyDrive/cse493s_hw1_runs runs

# Run experiments using the snippets in "Running experiments" above.
```

Iteration cycle: edit code locally → `git push` → `!git pull` in Colab → rerun.

---

## Autograder interface (`part_0_1_contract.py`)

- `load_model_and_tokenizer(checkpoint_dir)` → `(model, tokenizer)`
- `get_bos_token(tokenizer)` → `int` BOS token id
- `predict_answer(model, tokenizer, a, b, op, p)` → predicted `c`

A "checkpoint dir" is any directory containing a `ckpt.pt` written by
`train.py` (e.g. `runs/<run_name>/best`, `runs/<run_name>/final`).
