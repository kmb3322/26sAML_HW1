# Advanced ML HW 1

Yue Wu, Minbeom Kim

CSE 493S HW 1 (Spring 2026) — training infrastructure for the sanity check
and modular-arithmetic experiments (Parts 0 and 1).  See `ps.md` / `ps.pdf`
for the original problem set.

---

## Layout

```
.
├── model.py               # nano-GPT (starter; unchanged)
├── HW1_Part1.ipynb        # Run command for Google Colab
├── tokenizer_data.py      # residue & sanity tokenizers + data generation
├── train.py               # training loop + collate + masked CE + per-run plots
├── inference.py           # checkpoint loading, prediction, generation
├── part_0_1_contract.py   # autograder interface
├── configs/               # YAML config per experiment
├── runs/                  # checkpoints + metrics.csv
└── plots/                 # report-ready cross-run figures
```

In this assignment every batch contains fixed-length sequences (modular
task is always 7 tokens, sanity is a single sequence of length 6), so no
pad-attention masking is needed.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Configs

Every YAML file in `configs/` is a 1:1 mapping to the `RunConfig` dataclass
in `train.py`;
`python train.py --config configs/X.yaml`

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


A "checkpoint dir" is any directory containing a `ckpt.pt` written by
`train.py` (e.g. `runs/<run_name>/best`, `runs/<run_name>/final`).

