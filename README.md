# Advanced ML HW 1



## Layout

```
.
├── model.py                      # nano-GPT model (starter; unchanged)
├── tokenizer_data.py             # residue & sanity tokenizers + data gen
├── train_utils.py                # batch collation + masked cross-entropy
├── train.py                      # training entry point (sanity & modular)
├── inference.py                  # checkpoint loading, prediction, generation
├── plot_metrics.py               # CSV → loss/accuracy plots
├── part_0_1_contract.py          # autograder interface (Part 0 + Part 1)
├── configs/                      # YAML configs for each experiment
├── runs/                         # checkpoints + metrics.csv (git-ignored)
└── plots/                        # generated figures (git-ignored)
```

The original starter `model.py` is used as-is.  We added attention masking
for pad tokens only where it matters: in this assignment every batch
contains fixed-length sequences (modular task is always 7 tokens, sanity
is a single sequence of length 6), so no pad-attention masking is needed.

---

## Setup

```bash
pip install -r requirements.txt
```

Tested on PyTorch 2.x with both CPU and CUDA.

---

## Quick start

### 1. Sanity check (Part 0)

```bash
# Memorize "I love machine learning" with full-token loss.
python train.py --config configs/sanity.yaml

# Same sentence but mask the loss on the first 3 target positions; after
# training, prompting with "[BOS] I love" should yield "machine learning [EOS]".
python train.py --config configs/sanity_masked.yaml
```

Expected outcome: `train_loss → 0` and `train_acc = 1.0` (the full-string
greedy match is the reported "accuracy"). Both runs should converge in well
under 2000 steps on CPU.

Generation:

```bash
python inference.py --checkpoint_dir runs/sanity/final
python inference.py --checkpoint_dir runs/sanity_masked/final --sanity_prompt "I love"
```

### 2. Smoke test for modular task (CPU-friendly)

```bash
python train.py --config configs/smoke_modular.yaml
```

This runs only 1k steps so it finishes in under a minute on a laptop CPU.
Use it to confirm the pipeline before launching a real 100k-step run.

### 3. Addition / subtraction warmup (Part 1.2)

```bash
python train.py --config configs/add_p97_l1_seed0.yaml
python train.py --config configs/add_p97_l2_seed0.yaml
python train.py --config configs/add_p113_l1_seed0.yaml
python train.py --config configs/sub_p97_l1_seed0.yaml
# ... (4 more for the full p × n_layer × op grid)

# Random restarts: same config but different seed and out_dir.
python train.py --config configs/add_p97_l1_seed0.yaml --seed 1 \
    --out_dir runs/add_p97_l1_seed1
python train.py --config configs/add_p97_l1_seed0.yaml --seed 2 \
    --out_dir runs/add_p97_l1_seed2
```

Inference example:

```bash
python inference.py --checkpoint_dir runs/add_p97_l1_seed0/best \
    --a 12 --b 34 --op + --p 97
```

### 4. Grokking on modular division (Part 1.3)

```bash
python train.py --config configs/div_p97_grokking_seed0.yaml
```

The plotting helper has a built-in "grokking" mode that annotates the steps
at which train and test accuracy each cross 0.99 and reports the delay:

```bash
python plot_metrics.py --csv runs/div_p97_grokking_seed0/metrics.csv \
    --out_dir plots/div_p97_grokking_seed0 \
    --title "p=97, /, grokking" --grokking
```

### 5. Random-restart overlay

```bash
python plot_metrics.py \
    --csvs runs/add_p97_l1_seed0/metrics.csv \
           runs/add_p97_l1_seed1/metrics.csv \
           runs/add_p97_l1_seed2/metrics.csv \
    --labels seed0 seed1 seed2 \
    --out_dir plots/add_p97_l1_restarts \
    --title "Random restarts: p=97, +, 1-layer"
```

---

## Training infrastructure design

- **Tokenizer.** `ResidueTokenizer` represents each integer 0..max_p-1 as a
  single token, so every modular problem is exactly 7 tokens long
  (`[BOS, a, op, b, =, c, EOS]`).  We use `max_p=113` so the same vocabulary
  covers both `p=97` and `p=113` experiments.
- **Loss masking.** Targets in next-token training live in `ids[1:]`. The
  answer `c` is at target index `len(prompt) - 1` (i.e., the position right
  after `=`). Only that index is unmasked; everything else gets weight 0 in
  `masked_cross_entropy`.
- **Splits.** For `+`/`-` the dataset is `p²`. For `/` we exclude `b=0` (no
  modular inverse), giving `p(p−1)`. Default split is 50% train / 10% val /
  40% test, shuffled with a per-run seed.
- **Best checkpoint.** We track `val_acc` (modular) or full-string match
  (sanity) and save the best checkpoint accordingly. Test set is logged for
  plotting but never used for selection.
- **Configurability.** `train.py` accepts `--config <yaml>` and arbitrary
  CLI overrides; CLI > YAML > dataclass defaults.

---

## Local + Colab workflow

We develop locally and run heavy training on Google Colab.

```bash
# In Colab:
!git clone https://github.com/kmb3322/AML_HW1.git
%cd AML_HW1
!pip install -r requirements.txt

# Persist `runs/` to Drive so checkpoints survive runtime restarts.
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/cse493s_hw1_runs
!ln -sfn /content/drive/MyDrive/cse493s_hw1_runs runs

!python train.py --config configs/div_p97_grokking_seed0.yaml
```

Iteration cycle: edit code locally → `git push` → in Colab `!git pull` →
re-run the experiment.

---

## Part 0 and 1

The file `model.py` contains an implementation of a transformer model. The file `part_0_1_contract.py` contains some function signatures that would make autograding less painful for the TAs. 

## Part 2

The notebook `part_2_starter.ipynb` has code to load pretrained models, the AIME dataset, functions for evaluation and has unoptimized code for inference. We encourage you to write your own inference code to speed things up.



- `load_model_and_tokenizer(checkpoint_dir)` → `(model, tokenizer)`
- `get_bos_token(tokenizer)` → `int` BOS token id
- `predict_answer(model, tokenizer, a, b, op, p)` → predicted `c`
A "checkpoint dir" is any directory containing a `ckpt.pt` written by
`train.py` (e.g. `runs/<run_name>/best`, `runs/<run_name>/final`).