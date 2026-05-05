#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/dat.tt2/miniconda3/envs/dg/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
METHOD="${METHOD:-L_inv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-train_output/routing_diagnostics}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${METHOD}_env}"
ENVS="${ENVS:-0 1 2 3}"

DOMAIN_NAMES=(A C P S)
CLASS_NAMES=(dog elephant giraffe guitar horse house person)

RAW_FILES=()
METHOD_LABELS=()

for env in ${ENVS}; do
  out_dir="${OUTPUT_ROOT}/${OUTPUT_PREFIX}${env}"
  raw="${out_dir}/routing_raw.npz"

  if [[ ! -f "${raw}" ]]; then
    echo "Missing routing file: ${raw}" >&2
    echo "Run collect_routing.sh first, or set OUTPUT_PREFIX/ENVS to existing outputs." >&2
    exit 1
  fi

  echo "Plot env${env}: ${out_dir}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" domainbed/scripts/plot_routing_diagnostics.py \
    --routing_raw "${raw}" \
    --output_dir "${out_dir}/plots" \
    --domain_names "${DOMAIN_NAMES[@]}" \
    --class_names "${CLASS_NAMES[@]}"

  RAW_FILES+=("${raw}")
  METHOD_LABELS+=("${METHOD}")
done

table_dir="${OUTPUT_ROOT}/${METHOD}_all_envs_table"
echo "Make table: ${table_dir}"
"${PYTHON_BIN}" domainbed/scripts/make_routing_table.py \
  --routing_raw "${RAW_FILES[@]}" \
  --method "${METHOD_LABELS[@]}" \
  --output_dir "${table_dir}"
