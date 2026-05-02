#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/home/dat.tt2/miniconda3/envs/dg/bin/python}"
DATA_DIR="${DATA_DIR:-/mnt/disk1/backup_user/dat.tt2/data/cmnist}"
TOTAL_TRAIN_BUDGET="${TOTAL_TRAIN_BUDGET:-5600}"
OUTPUT_DIR="${OUTPUT_DIR:-train_outputs/cmnist20_gmoe_invmmd_target0_fixedbudget${TOTAL_TRAIN_BUDGET}_N2_to_N19_cuda0}"
CUDA_DEVICE="${CUDA_DEVICE:-4}"
SEED="${SEED:-0}"
TRIAL_SEED="${TRIAL_SEED:-0}"
HPARAMS_SEED="${HPARAMS_SEED:-0}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-}"
HPARAMS_JSON="${HPARAMS_JSON:-{\"cmnist_num_domains\":20,\"model\":\"deit_small_patch16_224\"}}"

for N in $(seq 8 19); do
  train_envs=$(seq -s ' ' 1 "$N")
  test_envs="0"
  base_examples_per_env=$((TOTAL_TRAIN_BUDGET / N))
  remainder_examples=$((TOTAL_TRAIN_BUDGET % N))
  max_train_examples_per_env=()
  for env_i in $(seq 0 $((N - 1))); do
    cap="$base_examples_per_env"
    if [ "$env_i" -lt "$remainder_examples" ]; then
      cap=$((cap + 1))
    fi
    max_train_examples_per_env+=("$cap")
  done

  echo "RUN N=$N train_envs=[$train_envs] test_envs=[$test_envs] total_train_budget=$TOTAL_TRAIN_BUDGET max_train_examples_per_env=[${max_train_examples_per_env[*]}]"

  cmd=(
    "$PYTHON_BIN" -m domainbed.scripts.train
    --data_dir "$DATA_DIR"
    --dataset ColoredMNIST
    --algorithm GMOE_InvMMD
    --train_envs $train_envs
    --test_envs $test_envs
    --eval_envs $test_envs
    --max_train_examples_per_env "${max_train_examples_per_env[@]}"
    --seed "$SEED"
    --trial_seed "$TRIAL_SEED"
    --hparams_seed "$HPARAMS_SEED"
    --batch_size "$BATCH_SIZE"
    --hparams "$HPARAMS_JSON"
    --checkpoint_freq "$CHECKPOINT_FREQ"
    --output_dir "$OUTPUT_DIR/N${N}_seed${SEED}"
    --skip_model_save
  )

  if [ -n "$STEPS" ]; then
    cmd+=(--steps "$STEPS")
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"
done
