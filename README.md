# Advanced ML HW 1

CSE 493S HW 1 (Spring 2026) — training infrastructure for the sanity check
and modular-arithmetic experiments (Parts 0 and 1).  See `ps.md` / `ps.pdf`
for the original problem set.

---

## Layout

```
.
├── model.py                      # nano-GPT (starter; unchanged)
├── tokenizer_data.py             # residue & sanity tokenizers + data gen
├── train_utils.py                # batch collation + masked cross-entropy
├── train.py                      # training entry point (sanity & modular)
├── inference.py                  # checkpoint loading, prediction, generation
├── plot_metrics.py               # generic CSV → loss/accuracy plot helpers
├── make_part1_plots.py           # report-ready aggregation across all runs
├── run_part1.sh                  # batch orchestrator (sanity/addsub/grokking/ablation)
├── part_0_1_contract.py          # autograder interface
├── configs/                      # YAML config per experiment
├── runs/                         # checkpoints + metrics.csv (git-ignored)
└── plots/                        # generated figures (git-ignored)
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

## Run all of Part 1 in one go (recommended)

```bash
# Idempotent — already-finished runs are skipped (detected via runs/<name>/final/ckpt.pt).
# Pass FORCE=1 to re-run from scratch.
bash run_part1.sh sanity     # Part 0/1 sanity (CPU; ~minutes)
bash run_part1.sh addsub     # Part 1.2: 8 main runs + 2 random restarts (GPU)
bash run_part1.sh grokking   # Part 1.3: division grokking (GPU)
bash run_part1.sh ablation   # Part 1.4: weight_decay × 3 + train_frac × 4 (GPU)
# or:
bash run_part1.sh all

# Aggregate everything into report-ready figures + summary CSV.
python make_part1_plots.py
```

Each `train.py` invocation also writes `loss.png` / `acc.png` next to its
own `metrics.csv`, so per-run figures are available immediately.

After `make_part1_plots.py` finishes you have:

```
plots/addsub/<run>/{loss.png, acc.png}
plots/addsub/p97_add_l1_restarts/{seeds_acc.png, seeds_loss.png}
plots/grokking/div_p97_grokking_seed0/{loss.png, acc.png, grokking_acc.png}
plots/ablation/wd/comparison_acc.png
plots/ablation/tf/comparison_acc.png
plots/ablation/summary.csv     # train_solved_step, test_solved_step, delay
```

---

## Manual / piecewise usage

Every YAML config can also be run directly:

```bash
python train.py --config configs/sanity.yaml
python train.py --config configs/add_p97_l1_seed0.yaml
python train.py --config configs/div_p97_grokking_seed0.yaml
```

CLI overrides take precedence over YAML, e.g.:

```bash
python train.py --config configs/add_p97_l1_seed0.yaml \
    --seed 1 --out_dir runs/add_p97_l1_seed1
```

Smoke test before any long run:

```bash
python train.py --config configs/smoke_modular.yaml   # ~1 min on CPU
```

Inference examples:

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

## Configs at a glance

```
configs/
  sanity.yaml                 # Part 0:    full-token loss
  sanity_masked.yaml          # Part 0:    prompt_len=3, train on suffix
  smoke_modular.yaml          # 1k-step smoke for the modular pipeline
  add_p{97,113}_l{1,2}_seed0.yaml,  sub_p{97,113}_l{1,2}_seed0.yaml
                              # Part 1.2:  8 main warmup runs
  div_p97_grokking_seed0.yaml # Part 1.3:  AdamW, wd=1.0, 100k steps
  abl_wd_{0.0,0.1,0.5}.yaml   # Part 1.4 ablation A: weight decay
  abl_tf_{0.30,0.40,0.60,0.80}.yaml
                              # Part 1.4 ablation B: train fraction
```

The `(p=97, +, 1-layer)` random-restart triple uses CLI overrides on
`add_p97_l1_seed0.yaml` (`--seed {0,1,2}`).

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

We develop locally and run heavy training on Colab.

```python
# In Colab:
!git clone https://github.com/kmb3322/AML_HW1.git
%cd AML_HW1
!pip install -q -r requirements.txt

# Persist runs/ to Drive so checkpoints survive runtime restarts.
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/cse493s_hw1_runs
!ln -sfn /content/drive/MyDrive/cse493s_hw1_runs runs

# Single command runs all of Part 1 (skips finished runs):
!bash run_part1.sh all
!python make_part1_plots.py
```

Iteration cycle: edit code locally → `git push` → in Colab `!git pull` →
rerun.

---

## Autograder interface (`part_0_1_contract.py`)

- `load_model_and_tokenizer(checkpoint_dir)` → `(model, tokenizer)`
- `get_bos_token(tokenizer)` → `int` BOS token id
- `predict_answer(model, tokenizer, a, b, op, p)` → predicted `c`

A "checkpoint dir" is any directory containing a `ckpt.pt` written by
`train.py` (e.g. `runs/<run_name>/best`, `runs/<run_name>/final`).
