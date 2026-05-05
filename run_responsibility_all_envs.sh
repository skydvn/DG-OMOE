#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/dat.tt2/miniconda3/envs/dg/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

METHOD="${METHOD:-L_inv_and_L_sp}"
DATASET="${DATASET:-PACS}"
ALGORITHM="${ALGORITHM:-MESSI}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-train_output/${METHOD}}"
if [[ -z "${CHECKPOINT_TEMPLATE:-}" ]]; then
  CHECKPOINT_TEMPLATE='pacs_gmoe_invmmd_env{env}_seed0/model.pkl'
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-train_output/responsibility_matrices}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${METHOD}_env}"

ENVS="${ENVS:-0 1 2 3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SPLIT="${SPLIT:-source_val}"
NORMALIZE="${NORMALIZE:-none}"

DATA_DIR_ARGS=()
if [[ -n "${DATA_DIR:-}" ]]; then
  DATA_DIR_ARGS=(--data_dir "${DATA_DIR}")
fi
DEVICE_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi
MAX_EXAMPLES_ARGS=()
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  MAX_EXAMPLES_ARGS=(--max_examples "${MAX_EXAMPLES}")
fi
ALPHA_ARGS=()
if [[ -n "${ALPHA:-}" ]]; then
  ALPHA_ARGS=(--alpha "${ALPHA}")
fi

if [[ "${DATASET}" == "PACS" ]]; then
  DOMAIN_NAMES=(A C P S)
elif [[ "${DATASET}" == "DomainNet" ]]; then
  DOMAIN_NAMES=(clipart infograph painting quickdraw real sketch)
else
  DOMAIN_NAMES=()
fi

for env in ${ENVS}; do
  env_token="{env}"
  checkpoint_rel="${CHECKPOINT_TEMPLATE//${env_token}/${env}}"
  checkpoint="${CHECKPOINT_ROOT}/${checkpoint_rel}"
  out_dir="${OUTPUT_ROOT}/${OUTPUT_PREFIX}${env}"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  echo "Collect responsibility env${env}: ${checkpoint}"
  "${PYTHON_BIN}" domainbed/scripts/collect_responsibility_matrices.py \
    --checkpoint "${checkpoint}" \
    --dataset "${DATASET}" \
    --algorithm "${ALGORITHM}" \
    --test_env "${env}" \
    --split "${SPLIT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --output_dir "${out_dir}" \
    "${DATA_DIR_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    "${MAX_EXAMPLES_ARGS[@]}" \
    "${ALPHA_ARGS[@]}"

  echo "Plot responsibility env${env}: ${out_dir}"
  if [[ ${#DOMAIN_NAMES[@]} -gt 0 ]]; then
    MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" domainbed/scripts/plot_responsibility_matrices.py \
      --input "${out_dir}/responsibility_raw.npz" \
      --dataset "${DATASET}" \
      --output "${out_dir}/responsibility_matrices.pdf" \
      --normalize "${NORMALIZE}" \
      --domain_names "${DOMAIN_NAMES[@]}"
  else
    MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" domainbed/scripts/plot_responsibility_matrices.py \
      --input "${out_dir}/responsibility_raw.npz" \
      --dataset "${DATASET}" \
      --output "${out_dir}/responsibility_matrices.pdf" \
      --normalize "${NORMALIZE}"
  fi
done

echo "Done."
