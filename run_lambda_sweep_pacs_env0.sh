#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/home/dat.tt2/miniconda3/envs/dg/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

DATA_DIR="${DATA_DIR:-./domainbed/data}"
DATASET="${DATASET:-PACS}"
ALGORITHM="${ALGORITHM:-GMOE_InvMMD}"
TEST_ENV="${TEST_ENV:-0}"
# SEEDS="${SEEDS:-0 1 2}"
SEEDS="${SEEDS:-0}"
LAMBDAS="${LAMBDAS:-0 1e-3 1e-2 1e-1 1 10}"

OUTPUT_ROOT="${OUTPUT_ROOT:-train_output/lambda_sweep/${DATASET}/${ALGORITHM}}"
RESULTS_DIR="${RESULTS_DIR:-${OUTPUT_ROOT}/results_env${TEST_ENV}}"

BATCH_SIZE="${BATCH_SIZE:-}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-}"
STEPS="${STEPS:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-}"
DISTANCE="${DISTANCE:-mmd}"
CHANCE="${CHANCE:-0.3333333333}"

SKIP_TRAIN="${SKIP_TRAIN:-1}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"

run_python() {
  if [[ -n "${CUDA_DEVICE}" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" "$@"
  else
    MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" "$@"
  fi
}

lambda_dir_token() {
  printf '%s' "$1" | sed 's/-/m/g'
}

train_one() {
  local seed="$1"
  local lam="$2"
  local lam_token
  lam_token="$(lambda_dir_token "${lam}")"
  local run_dir="${OUTPUT_ROOT}/env${TEST_ENV}_seed${seed}_lam${lam_token}"

  if [[ -f "${run_dir}/model.pkl" ]]; then
    echo "[train] skip existing ${run_dir}/model.pkl"
    return
  fi

  mkdir -p "${run_dir}"
  echo "[train] env=${TEST_ENV} seed=${seed} lambda_inv=${lam} -> ${run_dir}"

  cmd=(
    -m domainbed.scripts.train
    --data_dir "${DATA_DIR}"
    --dataset "${DATASET}"
    --algorithm "${ALGORITHM}"
    --test_envs "${TEST_ENV}"
    --seed "${seed}"
    --trial_seed "${seed}"
    --hparams "{\"lambda_inv\": ${lam}}"
    --output_dir "${run_dir}"
  )

  if [[ -n "${BATCH_SIZE}" ]]; then
    cmd+=(--batch_size "${BATCH_SIZE}")
  fi
  if [[ -n "${CHECKPOINT_FREQ}" ]]; then
    cmd+=(--checkpoint_freq "${CHECKPOINT_FREQ}")
  fi
  if [[ -n "${STEPS}" ]]; then
    cmd+=(--steps "${STEPS}")
  fi

  run_python "${cmd[@]}"
}

collect_one() {
  local run_dir="$1"

  if [[ ! -f "${run_dir}/model.pkl" ]]; then
    echo "[collect] missing checkpoint: ${run_dir}/model.pkl" >&2
    exit 1
  fi

  if [[ -f "${run_dir}/features_source_val.npz" && -f "${run_dir}/features_target.npz" ]]; then
    echo "[collect] skip existing features in ${run_dir}"
    return
  fi

  echo "[collect] ${run_dir}"
  run_python domainbed/scripts/collect_lambda_sweep_features.py \
    --checkpoint "${run_dir}/model.pkl" \
    --output_dir "${run_dir}" \
    --num_workers "${NUM_WORKERS}"
}

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  for seed in ${SEEDS}; do
    for lam in ${LAMBDAS}; do
      train_one "${seed}" "${lam}"
    done
  done
fi

if [[ "${SKIP_COLLECT}" != "1" ]]; then
  run_dirs=()
  for seed in ${SEEDS}; do
    for lam in ${LAMBDAS}; do
      lam_token="$(lambda_dir_token "${lam}")"
      run_dirs+=("${OUTPUT_ROOT}/env${TEST_ENV}_seed${seed}_lam${lam_token}")
    done
  done

  for run_dir in "${run_dirs[@]}"; do
    collect_one "${run_dir}"
  done
fi

run_dirs=()
for seed in ${SEEDS}; do
  for lam in ${LAMBDAS}; do
    lam_token="$(lambda_dir_token "${lam}")"
    run_dirs+=("${OUTPUT_ROOT}/env${TEST_ENV}_seed${seed}_lam${lam_token}")
  done
done

echo "[plot] ${RESULTS_DIR}"
run_python domainbed/scripts/plot_lambda_sweep.py \
  --run_dirs "${run_dirs[@]}" \
  --output_dir "${RESULTS_DIR}" \
  --distance "${DISTANCE}" \
  --chance "${CHANCE}"

echo "Done."
echo "CSV: ${RESULTS_DIR}/lambda_sweep_results.csv"
echo "TeX: ${RESULTS_DIR}/lambda_sweep_results.tex"
echo "PDF: ${RESULTS_DIR}/lambda_sweep.pdf"
