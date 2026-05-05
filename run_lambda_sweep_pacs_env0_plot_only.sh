#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/home/dat.tt2/miniconda3/envs/dg/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

DATASET="${DATASET:-PACS}"
ALGORITHM="${ALGORITHM:-GMOE_InvMMD}"
TEST_ENV="${TEST_ENV:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-train_output/lambda_sweep/${DATASET}/${ALGORITHM}}"
RESULTS_DIR="${RESULTS_DIR:-${OUTPUT_ROOT}/results_env${TEST_ENV}}"

CHANCE="${CHANCE:-0.3333333333}"
CSV="${CSV:-}"

run_python() {
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" "$@"
}

if [[ -z "${CSV}" ]]; then
  if [[ -f "${RESULTS_DIR}/lambda_sweep_result.csv" ]]; then
    CSV="${RESULTS_DIR}/lambda_sweep_result.csv"
  elif [[ -f "${RESULTS_DIR}/lambda_sweep_results.csv" ]]; then
    CSV="${RESULTS_DIR}/lambda_sweep_results.csv"
  else
    echo "[plot] missing CSV file." >&2
    echo "[plot] expected one of:" >&2
    echo "  ${RESULTS_DIR}/lambda_sweep_result.csv" >&2
    echo "  ${RESULTS_DIR}/lambda_sweep_results.csv" >&2
    echo "[plot] or pass CSV=/path/to/lambda_sweep_result.csv" >&2
    exit 1
  fi
fi

echo "[plot] from CSV: ${CSV}"
run_python domainbed/scripts/plot_lambda_sweep.py \
  --csv "${CSV}" \
  --output_dir "${RESULTS_DIR}" \
  --chance "${CHANCE}"

echo "Done."
echo "CSV: ${RESULTS_DIR}/lambda_sweep_results.csv"
echo "TeX: ${RESULTS_DIR}/lambda_sweep_results.tex"
echo "PDF: ${RESULTS_DIR}/lambda_sweep.pdf"
