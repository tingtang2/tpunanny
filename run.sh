#!/bin/bash
set -euo pipefail

READY_FILE="/tmp/loss_spikes_setup_ready"
MAX_WAIT_SECONDS=1800
SLEEP_SECONDS=10
waited=0


until [[ -f "$READY_FILE" ]]; do
  if (( waited >= MAX_WAIT_SECONDS )); then
    echo "Timed out waiting for setup marker: $READY_FILE"
    exit 1
  fi
  echo "Waiting for setup to complete... (${waited}s/${MAX_WAIT_SECONDS}s)"
  sleep "$SLEEP_SECONDS"
  waited=$((waited + SLEEP_SECONDS))
done

echo "Setup marker detected. Starting training."

SEED="${SEED:-0}"
LR_TAG="${LR_TAG:-default}"
LR_ARG="${LR_ARG:-}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-exp_gpt3xl}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WANDB_MODE="${WANDB_MODE:-online}"
CONFIG_NAME="${CONFIG_NAME:-wortsman_default}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-10000}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-gs://demand-v4-checkpoint-storage/picodo_ckpts}"
USE_CHINCHILLA="${USE_CHINCHILLA:-false}"

RUN_NAME="${RUN_NAME_PREFIX}_seed${SEED}_lr${LR_TAG}"
CHECKPOINT_BUCKET_PATH="${CHECKPOINT_ROOT}/${RUN_NAME}"
SESSION_NAME="picodo_train_seed${SEED}"
LOG_DIR="$HOME/loss-spikes-project/picodo/train_logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

mkdir -p "$LOG_DIR"

cd "$HOME/loss-spikes-project"
source env-loss-spikes/bin/activate
cd picodo

git pull

# Ensure only one trainer session is active per worker.
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true



TRAIN_ARGS=(
  "+model=gpt3xl"
  "run_name=${RUN_NAME}"
  "checkpoint.turn_on=true"
  "+checkpoint.gcp_bucket=${CHECKPOINT_BUCKET_PATH}"
  "checkpoint.start_step=null"
  "checkpoint.checkpoint_steps=null"
  "checkpoint.checkpoint_every_steps=${CHECKPOINT_FREQ}"
  "opt.batch_size=${BATCH_SIZE}"
  "seed=${SEED}"
  "wandb_mode=${WANDB_MODE}"
)

if [[ "${RUN_NAME_PREFIX,,}" == *"gpt3xl"* ]]; then
  TRAIN_ARGS+=("+dataset=fw_gpt2_100B")
else
  TRAIN_ARGS+=("+dataset=fw_gpt2")
fi

if [[ -n "$LR_ARG" ]]; then
  TRAIN_ARGS+=("$LR_ARG")
fi

if [[ "${USE_CHINCHILLA,,}" == "true" ]]; then
  TRAIN_ARGS+=("num_tokens_train=null")
fi

TRAIN_ARGS+=("--config-name=${CONFIG_NAME}")

printf -v ESCAPED_TRAIN_ARGS '%q ' "${TRAIN_ARGS[@]}"
TMUX_CMD="python3 -u main.py ${ESCAPED_TRAIN_ARGS} >> \"$LOG_FILE\" 2>&1"

tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"
echo "Started tmux session '$SESSION_NAME'."
echo "Run name: $RUN_NAME"
echo "Checkpoint path: $CHECKPOINT_BUCKET_PATH"
echo "Logs: $LOG_FILE"
