#!/usr/bin/env bash
set -euo pipefail

# Run the objective-ablation table:
#   MESSI, w/o L_ssi, w/o L_sp, w/o L_bal, w/o L_div,
#   w/o L_sp,L_bal,L_div.
#
# Defaults target the PACS leave-one-out table. Override DATASET/TEST_ENVS to
# reuse the same grid for other datasets, for example:
#   DATASET=WILDSIWildCam TEST_ENVS="1 2 3 4" SEEDS=3 bash scripts/phase1/run_objective_ablation_table.sh
#
# If a dataset needs multiple held-out envs in a single run, pass comma-separated
# groups:
#   TEST_ENV_GROUPS="1,2,3,4" bash scripts/phase1/run_objective_ablation_table.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/domainbed/data}"
DATASET="${DATASET:-PACS}"
ALGORITHM="${ALGORITHM:-GMOE_InvMMD}"
# TEST_ENVS="${TEST_ENVS:-0 1 2 3}"
TEST_ENVS="${TEST_ENVS:-0 2 3}"
SEEDS="${SEEDS:-1}"
STEPS="${STEPS:-5000}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-500}"
WANDB_MODE="${WANDB_MODE:-disabled}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/train_output/objective_ablation_${DATASET}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
ROUTING_JS_MIN_COUNT="${ROUTING_JS_MIN_COUNT:-5}"
DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-64}"
DIAG_NUM_WORKERS="${DIAG_NUM_WORKERS:-0}"
RESUME_DONE="${RESUME_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

BASE_HPARAMS='{"data_augmentation":true,"batch_size":32,"model":"deit_small_patch16_224","lambda_inv":0.01,"lambda_sp":0.01,"lambda_bal":0.01,"lambda_div":0.02,"alpha":4.0}'

variant_hparams() {
    case "$1" in
        messi)
            printf '%s' "$BASE_HPARAMS"
            ;;
        without_l_ssi)
            printf '%s' '{"data_augmentation":true,"batch_size":32,"model":"deit_small_patch16_224","lambda_inv":0.0,"lambda_sp":0.01,"lambda_bal":0.01,"lambda_div":0.02,"alpha":4.0}'
            ;;
        without_l_sp)
            printf '%s' '{"data_augmentation":true,"batch_size":32,"model":"deit_small_patch16_224","lambda_inv":0.01,"lambda_sp":0.0,"lambda_bal":0.01,"lambda_div":0.02,"alpha":4.0}'
            ;;
        without_l_bal)
            printf '%s' '{"data_augmentation":true,"batch_size":32,"model":"deit_small_patch16_224","lambda_inv":0.01,"lambda_sp":0.01,"lambda_bal":0.0,"lambda_div":0.02,"alpha":4.0}'
            ;;
        without_l_div)
            printf '%s' '{"data_augmentation":true,"batch_size":32,"model":"deit_small_patch16_224","lambda_inv":0.01,"lambda_sp":0.01,"lambda_bal":0.01,"lambda_div":0.0,"alpha":4.0}'
            ;;
        without_specializers)
            printf '%s' '{"data_augmentation":true,"batch_size":32,"model":"deit_small_patch16_224","lambda_inv":0.01,"lambda_sp":0.0,"lambda_bal":0.0,"lambda_div":0.0,"alpha":4.0}'
            ;;
        *)
            echo "Unknown variant: $1" >&2
            return 1
            ;;
    esac
}

variant_label() {
    case "$1" in
        messi) printf '%s' 'MESSI' ;;
        without_l_ssi) printf '%s' 'w/o L_ssi' ;;
        without_l_sp) printf '%s' 'w/o L_sp' ;;
        without_l_bal) printf '%s' 'w/o L_bal' ;;
        without_l_div) printf '%s' 'w/o L_div' ;;
        without_specializers) printf '%s' 'w/o L_sp,L_bal,L_div' ;;
        *) printf '%s' "$1" ;;
    esac
}

env_groups() {
    if [ -n "${TEST_ENV_GROUPS:-}" ]; then
        printf '%s\n' "$TEST_ENV_GROUPS"
    else
        for env in $TEST_ENVS; do
            printf '%s\n' "$env"
        done
    fi
}

env_arg_from_group() {
    printf '%s' "${1//,/ }"
}

env_slug_from_group() {
    printf 'env%s' "${1//,/-}"
}

run_cmd() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

VARIANTS=(
    messi
    without_l_ssi
    without_l_sp
    without_l_bal
    without_l_div
    without_specializers
)

echo "============================================================"
echo "Objective ablation table"
echo "  dataset         = ${DATASET}"
echo "  algorithm       = ${ALGORITHM}"
echo "  test env groups = ${TEST_ENV_GROUPS:-${TEST_ENVS}}"
echo "  seeds           = ${SEEDS}"
echo "  steps           = ${STEPS}"
echo "  checkpoint_freq = ${CHECKPOINT_FREQ}"
echo "  output_root     = ${OUTPUT_ROOT}"
echo "  wandb_mode      = ${WANDB_MODE}"
echo "============================================================"

for variant in "${VARIANTS[@]}"; do
    hparams="$(variant_hparams "$variant")"
    label="$(variant_label "$variant")"

    while IFS= read -r group; do
        [ -n "$group" ] || continue
        test_env_args="$(env_arg_from_group "$group")"
        env_slug="$(env_slug_from_group "$group")"

        for seed in $SEEDS; do
            run_name="${variant}_${env_slug}_seed${seed}"
            run_dir="${OUTPUT_ROOT}/${run_name}"
            log_file="${LOG_DIR}/${run_name}.log"

            if [ "$RESUME_DONE" = "1" ] && [ -f "${run_dir}/done" ]; then
                echo "[skip] ${run_name} already has done file"
                continue
            fi

            echo "------------------------------------------------------------"
            echo "[run] variant=${label} test_envs=${test_env_args} seed=${seed}"
            echo "      output=${run_dir}"
            echo "      log=${log_file}"

            cmd=(
                "$PYTHON_BIN" -u -m domainbed.scripts.train
                --dataset "$DATASET"
                --algorithm "$ALGORITHM"
                --test_envs $test_env_args
                --output_dir "$run_dir"
                --data_dir "$DATA_DIR"
                --hparams "$hparams"
                --steps "$STEPS"
                --checkpoint_freq "$CHECKPOINT_FREQ"
                --seed "$seed"
                --trial_seed "$seed"
            )

            if [ -n "${CUDA_DEVICE:-}" ]; then
                if [ "$DRY_RUN" = "1" ]; then
                    echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} WANDB_MODE=${WANDB_MODE}"
                    run_cmd "${cmd[@]}"
                else
                    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
                    PYTHONUNBUFFERED=1 \
                    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                    WANDB_MODE="$WANDB_MODE" \
                    "${cmd[@]}" $EXTRA_ARGS 2>&1 | tee "$log_file"
                fi
            else
                if [ "$DRY_RUN" = "1" ]; then
                    echo "WANDB_MODE=${WANDB_MODE}"
                    run_cmd "${cmd[@]}"
                else
                    PYTHONUNBUFFERED=1 \
                    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                    WANDB_MODE="$WANDB_MODE" \
                    "${cmd[@]}" $EXTRA_ARGS 2>&1 | tee "$log_file"
                fi
            fi
        done
    done < <(env_groups)
done

if [ "$RUN_DIAGNOSTICS" = "1" ]; then
    if find "$OUTPUT_ROOT" -maxdepth 2 -name model.pkl -print -quit | grep -q .; then
        echo "------------------------------------------------------------"
        echo "[diagnostics] recomputing source-val routing diagnostics"
        run_cmd "$PYTHON_BIN" scripts/phase1/recompute_source_val_diagnostics.py \
            --output_dir "$OUTPUT_ROOT" \
            --data_dir "$DATA_DIR" \
            --batch_size "$DIAG_BATCH_SIZE" \
            --num_workers "$DIAG_NUM_WORKERS" \
            --routing_js_min_count "$ROUTING_JS_MIN_COUNT"
    else
        echo "[diagnostics] no checkpoints found under ${OUTPUT_ROOT}; skipping"
    fi
fi

echo "------------------------------------------------------------"
echo "[summary] writing ablation table"
run_cmd "$PYTHON_BIN" scripts/phase1/summarize_objective_ablation_table.py \
    --output_dir "$OUTPUT_ROOT" \
    --dataset "$DATASET"

echo "============================================================"
echo "Done. Table outputs are under ${OUTPUT_ROOT}"
echo "============================================================"
