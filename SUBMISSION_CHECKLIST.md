# HW1 Submission Checklist

## Submit to Gradescope

Include code, report, environment definitions, and artifacts. If the full
artifact folder is too large for Gradescope, submit the code/report and provide
a private repo or Drive link according to the course instructions.

## Code

Submit these source files:

- `model.py`
- `tokenizer_data.py`
- `train.py`
- `inference.py`
- `part_0_1_contract.py`
- `HW1_Part1.ipynb`
- `part_2_starter.ipynb` or the completed Part 2 notebook/script
- `configs/`
- `requirements.txt`
- `README.md`

Do not rely on untracked local-only files.

## Part 1 Artifacts

Submit or link the full `cse493s_hw1_runs/` folder, or at least:

- All `metrics.csv` files for the main Part 1 runs.
- Per-run `acc.png` and `loss.png` plots.
- Combined report plots under `cse493s_hw1_runs/plots/`.
- The final random-restart checkpoint requested by the course staff:
  `cse493s_hw1_runs/add_p97_l1_seed2/final/ckpt.pt`.
- The grokking final checkpoint:
  `cse493s_hw1_runs/div_p97_grokking_seed0/final/ckpt.pt`.

The current `cse493s_hw1_runs/` folder also includes `best`, `latest`, and
`final` checkpoints for every run.

## Part 2 Artifacts Still Needed

The current artifact folder does not contain Part 2 outputs. Before final
submission, add:

- A completed Part 2 notebook or script.
- Raw generations/completions for AIME 2024.
- CSVs with exact and flexible extraction results.
- Thinking-length histogram.
- Sequential scaling plots.
- Parallel scaling plots.
- Saved qualitative reasoning traces.
- Results for two parallel-scaling improvements.

## Report

Current draft:

- `REPORT.md`

For final submission, convert to PDF if possible after adding Part 2 results.
Markdown is acceptable per the assignment, but PDF is easier for readability.

## Suggested Packaging

Small code package:

```bash
zip -r hw1_code_report.zip \
  model.py tokenizer_data.py train.py inference.py part_0_1_contract.py \
  configs requirements.txt README.md REPORT.md SUBMISSION_CHECKLIST.md \
  HW1_Part1.ipynb part_2_starter.ipynb
```

Artifacts package:

```bash
zip -r hw1_part1_artifacts.zip cse493s_hw1_runs
```

If submitting one package, include both code and artifacts, but avoid including
`.git`, `__pycache__`, `.DS_Store`, or local virtual environments.
