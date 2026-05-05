#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/dat.tt2/miniconda3/envs/dg/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

METHOD="${METHOD:-gmoe_baseline}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-train_output/${METHOD}}"
if [[ -z "${CHECKPOINT_TEMPLATE:-}" ]]; then
  CHECKPOINT_TEMPLATE='pacs_gmoe_env{env}_seed0/model.pkl'
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-train_output/routing_diagnostics}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${METHOD}_env}"

ENVS="${ENVS:-0 1 2 3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SPLIT="${SPLIT:-out}"
DEVICE_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi
MAX_EXAMPLES_ARGS=()
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  MAX_EXAMPLES_ARGS=(--max_examples "${MAX_EXAMPLES}")
fi

DOMAIN_NAMES=(A C P S)
CLASS_NAMES=(dog elephant giraffe guitar horse house person)

RAW_FILES=()
METHOD_LABELS=()

for env in ${ENVS}; do
  env_token="{env}"
  checkpoint_rel="${CHECKPOINT_TEMPLATE//${env_token}/${env}}"
  checkpoint="${CHECKPOINT_ROOT}/${checkpoint_rel}"
  out_dir="${OUTPUT_ROOT}/${OUTPUT_PREFIX}${env}"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  echo "Collect env${env}: ${checkpoint}"
  "${PYTHON_BIN}" domainbed/scripts/collect_routing_diagnostics.py \
    --checkpoint "${checkpoint}" \
    --output_dir "${out_dir}" \
    --split "${SPLIT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    "${DEVICE_ARGS[@]}" \
    "${MAX_EXAMPLES_ARGS[@]}"

  echo "Plot env${env}: ${out_dir}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" domainbed/scripts/plot_routing_diagnostics.py \
    --routing_raw "${out_dir}/routing_raw.npz" \
    --output_dir "${out_dir}/plots" \
    --domain_names "${DOMAIN_NAMES[@]}" \
    --class_names "${CLASS_NAMES[@]}"

  RAW_FILES+=("${out_dir}/routing_raw.npz")
  METHOD_LABELS+=("${METHOD}")
done

table_dir="${OUTPUT_ROOT}/${METHOD}_all_envs_table"
echo "Make table: ${table_dir}"
"${PYTHON_BIN}" domainbed/scripts/make_routing_table.py \
  --routing_raw "${RAW_FILES[@]}" \
  --method "${METHOD_LABELS[@]}" \
  --output_dir "${table_dir}"

echo "Done."
