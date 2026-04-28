#!/usr/bin/env bash
# Batch orchestrator for Part 1 experiments.
#
# Usage:
#   bash run_part1.sh [phase]
#
# Phases:
#   sanity     — Part 0/1 sanity checks (CPU-friendly)
#   addsub     — Part 1.2 addition/subtraction (8 main + 2 random-restart seeds)
#   grokking   — Part 1.3 division grokking (single run, p=97, AdamW wd=1.0)
#   ablation   — Part 1.4 ablations (weight_decay × 3 + train_frac × 4)
#   all        — sanity → addsub → grokking → ablation
#
# Already-finished runs are skipped (detected via runs/<name>/final/ckpt.pt).
# Pass FORCE=1 to re-run everything: FORCE=1 bash run_part1.sh all

set -euo pipefail

PHASE="${1:-all}"
FORCE="${FORCE:-0}"

run_cfg() {
    # run_cfg <config_path> [extra args...]
    local cfg="$1"; shift || true
    local out_dir
    out_dir="$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$cfg'))['out_dir'])")"

    # Allow CLI overrides to change out_dir.  Walk through extra args looking for --out_dir.
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == "--out_dir" ]]; then
            out_dir="${args[i+1]}"
        fi
    done

    if [[ -f "${out_dir}/final/ckpt.pt" && "${FORCE}" != "1" ]]; then
        echo "[skip] ${out_dir} already has final/ckpt.pt (set FORCE=1 to re-run)."
        return 0
    fi

    echo "============================================================"
    echo "[run] ${cfg}  -> ${out_dir}  ${args[*]:-}"
    echo "============================================================"
    python3 train.py --config "${cfg}" "${args[@]:-}"
}

phase_sanity() {
    run_cfg configs/sanity.yaml
    run_cfg configs/sanity_masked.yaml
}

phase_addsub() {
    # 2 ops × 2 p × 2 layers = 8 main runs
    run_cfg configs/add_p97_l1_seed0.yaml
    run_cfg configs/add_p97_l2_seed0.yaml
    run_cfg configs/add_p113_l1_seed0.yaml
    run_cfg configs/add_p113_l2_seed0.yaml
    run_cfg configs/sub_p97_l1_seed0.yaml
    run_cfg configs/sub_p97_l2_seed0.yaml
    run_cfg configs/sub_p113_l1_seed0.yaml
    run_cfg configs/sub_p113_l2_seed0.yaml

    # Random restarts for the required (p=97, +, 1-layer) row.
    run_cfg configs/add_p97_l1_seed0.yaml --seed 1 --out_dir runs/add_p97_l1_seed1
    run_cfg configs/add_p97_l1_seed0.yaml --seed 2 --out_dir runs/add_p97_l1_seed2
}

phase_grokking() {
    run_cfg configs/div_p97_grokking_seed0.yaml
}

phase_ablation() {
    # Ablation A: weight_decay (baseline wd=1.0 reuses the grokking run).
    run_cfg configs/abl_wd_0.0.yaml
    run_cfg configs/abl_wd_0.1.yaml
    run_cfg configs/abl_wd_0.5.yaml

    # Ablation B: train_frac (baseline 0.50 reuses the grokking run).
    run_cfg configs/abl_tf_0.30.yaml
    run_cfg configs/abl_tf_0.40.yaml
    run_cfg configs/abl_tf_0.60.yaml
    run_cfg configs/abl_tf_0.80.yaml
}

case "${PHASE}" in
    sanity)   phase_sanity ;;
    addsub)   phase_addsub ;;
    grokking) phase_grokking ;;
    ablation) phase_ablation ;;
    all)      phase_sanity; phase_addsub; phase_grokking; phase_ablation ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo "Use one of: sanity | addsub | grokking | ablation | all"
        exit 2
        ;;
esac

echo
echo "Done with phase: ${PHASE}"
echo "Now run:  python3 make_part1_plots.py"
