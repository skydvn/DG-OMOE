#!/usr/bin/env bash
set -euo pipefail

# PACS leave-one-domain-out training for GMOE_InvMMD.
#
# Usage:
#   bash scripts/phase1/train_pacs_gmoe_invmmd.sh
#   TEST_ENVS="0 1 2 3" CUDA_DEVICE=0 STEPS=5000 bash scripts/phase1/train_pacs_gmoe_invmmd.sh
#   TEST_ENV=2 CUDA_DEVICE=0 STEPS=5000 bash scripts/phase1/train_pacs_gmoe_invmmd.sh
#
# PACS env indices:
#   0=A, 1=C, 2=P, 3=S

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/dat.tt2/miniconda3/envs/dg/bin/python}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/domainbed/data}"
# TEST_ENVS="${TEST_ENVS:-${TEST_ENV:-0 1 2 3}}"
TEST_ENVS="${TEST_ENVS:-${TEST_ENV:-1 2}}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-5000}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-300}"
SEED="${SEED:-0}"
TRIAL_SEED="${TRIAL_SEED:-0}"
HPARAMS_SEED="${HPARAMS_SEED:-0}"
WANDB_MODE="${WANDB_MODE:-disabled}"
OUTPUT_BASE="${OUTPUT_BASE:-train_output/phase1}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
fi

if [ ! -d "${DATA_DIR}/PACS" ]; then
    echo "ERROR: PACS dataset not found at ${DATA_DIR}/PACS"
    echo "Hint: DATA_DIR must point to the parent folder containing PACS."
    exit 1
fi

HPARAMS_JSON="${HPARAMS_JSON:-{\"model\":\"deit_small_patch16_224\",\"num_experts\":6,\"gate_k\":1,\"expert_depth\":2,\"mlp_ratio\":4.0,\"lambda_inv\":0.1,\"alpha\":4.0}}"

echo "============================================================"
echo "PACS GMOE_InvMMD training"
echo "  test_envs       = ${TEST_ENVS}  (0=A, 1=C, 2=P, 3=S)"
echo "  data_dir        = ${DATA_DIR}"
echo "  output_base     = ${OUTPUT_BASE}"
echo "  python          = ${PYTHON_BIN}"
echo "  cuda_device     = ${CUDA_DEVICE}"
echo "  batch_size      = ${BATCH_SIZE}"
echo "  steps           = ${STEPS}"
echo "  checkpoint_freq = ${CHECKPOINT_FREQ}"
echo "  hparams         = ${HPARAMS_JSON}"
echo "  wandb_mode      = ${WANDB_MODE}"
echo "============================================================"

cd "$REPO_ROOT"

read -r -a TEST_ENV_ARRAY <<< "$TEST_ENVS"
NUM_TEST_ENVS="${#TEST_ENV_ARRAY[@]}"

for TEST_ENV in "${TEST_ENV_ARRAY[@]}"; do
    case "$TEST_ENV" in
        0|1|2|3) ;;
        *)
            echo "ERROR: invalid TEST_ENV=${TEST_ENV}; expected one of 0 1 2 3"
            exit 1
            ;;
    esac

    if [ -n "${OUTPUT_DIR:-}" ] && [ "$NUM_TEST_ENVS" -eq 1 ]; then
        RUN_OUTPUT_DIR="$OUTPUT_DIR"
    else
        RUN_OUTPUT_DIR="${OUTPUT_BASE}/pacs_gmoe_invmmd_env${TEST_ENV}_seed${SEED}"
    fi

    echo "============================================================"
    echo "Running PACS env ${TEST_ENV} -> ${RUN_OUTPUT_DIR}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_MODE="$WANDB_MODE" \
    "$PYTHON_BIN" -u -m domainbed.scripts.train \
        --dataset PACS \
        --algorithm GMOE_InvMMD \
        --test_envs "$TEST_ENV" \
        --data_dir "$DATA_DIR" \
        --output_dir "$RUN_OUTPUT_DIR" \
        --hparams "$HPARAMS_JSON" \
        --batch_size "$BATCH_SIZE" \
        --steps "$STEPS" \
        --checkpoint_freq "$CHECKPOINT_FREQ" \
        --hparams_seed "$HPARAMS_SEED" \
        --seed "$SEED" \
        --trial_seed "$TRIAL_SEED"

    echo "Done env ${TEST_ENV}. Checkpoint: ${RUN_OUTPUT_DIR}/model.pkl"
    echo "Routing diagnostics: ${RUN_OUTPUT_DIR}/routing_diagnostics.jsonl"
    echo "Full train log: ${RUN_OUTPUT_DIR}/out.txt"
done

echo "============================================================"
echo "Done all requested PACS envs: ${TEST_ENVS}"
echo "============================================================"
