#!/bin/bash
set -euo pipefail

USER_NAME="tingchen"
USER_HOME="/home/tingchen"
PROJECT_DIR="$USER_HOME/loss-spikes-project"
REPO_DIR="$PROJECT_DIR/picodo"
VENV_DIR="$PROJECT_DIR/env-loss-spikes"
SETUP_MARKER="$PROJECT_DIR/.tpunanny_setup_done"

export HOME="$USER_HOME"
export USER="$USER_NAME"
export LOGNAME="$USER_NAME"
mkdir -p "$PROJECT_DIR"

SEED="${SEED:-0}"
LR_TAG="${LR_TAG:-default}"
LR_ARG="${LR_ARG:-}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-exp_gpt3xl}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WANDB_MODE="${WANDB_MODE:-online}"
CONFIG_NAME="${CONFIG_NAME:-wortsman_default}"
EVAL_FREQ="${EVAL_FREQ:-100}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-10000}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-gs://demand-v4-checkpoint-storage/picodo_ckpts}"
USE_CHINCHILLA="${USE_CHINCHILLA:-false}"

RUN_NAME="${RUN_NAME_PREFIX}_seed${SEED}_lr${LR_TAG}"
CHECKPOINT_BUCKET_PATH="${CHECKPOINT_ROOT}/${RUN_NAME}"
SESSION_NAME="picodo_train_seed${SEED}"
LOG_DIR="$REPO_DIR/train_logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
STATUS_DIR="$REPO_DIR/train_status"
DONE_MARKER="$STATUS_DIR/${RUN_NAME}.done"
FAILED_MARKER="$STATUS_DIR/${RUN_NAME}.failed"
STATUS_FILE="$STATUS_DIR/${RUN_NAME}.exitcode"

wait_for_apt_lock() {
  while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || \
        sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
    echo "Waiting for apt/dpkg lock..."
    sleep 5
  done
}

apt_with_retry() {
  local cmd="$1"
  local attempts=12
  local delay=10
  local i
  for ((i=1; i<=attempts; i++)); do
    wait_for_apt_lock
    if eval "$cmd"; then
      return 0
    fi
    if (( i < attempts )); then
      echo "apt command failed (attempt $i/$attempts). Retrying in ${delay}s..."
      sleep "$delay"
    fi
  done
  echo "apt command failed after ${attempts} attempts: $cmd"
  return 1
}

ensure_setup() {
  if [[ -f "$SETUP_MARKER" ]]; then
    echo "Setup marker exists; skipping bootstrap."
    return
  fi

  apt_with_retry "sudo apt-get update"
  apt_with_retry "sudo apt-get install -y libopenblas-dev tmux golang"

  if [[ ! -x "$USER_HOME/.local/bin/uv" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  source "$USER_HOME/.local/bin/env"

  if [[ ! -d "$VENV_DIR" ]]; then
    uv venv "$VENV_DIR" --python=3.11
  fi
  source "$VENV_DIR/bin/activate"

  if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone https://github.com/tingtang2/picodo.git "$REPO_DIR"
  fi

  cd "$REPO_DIR"
  uv pip install -r requirements.txt

  git config --global user.email "zc2666@columbia.edu"
  git config --global user.name "Ting Chen"

  if [[ -n "${WANDB_TOKEN:-}" ]]; then
    wandb login "$WANDB_TOKEN"
  fi

  BUCKET="demand-v4-checkpoint-storage"
  PREFIX="picodo_ckpts"
  PROJECT_ID="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/project-id)"
  SA_EMAIL="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email)"
  INSTANCE_NAME="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name)"
  WORKER_ID="${INSTANCE_NAME##*-}"

  if [[ "$WORKER_ID" == "0" ]]; then
    gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
      --member="serviceAccount:${SA_EMAIL}" \
      --role="roles/storage.objectAdmin" \
      --project="${PROJECT_ID}" || true

    echo "ok $(date -u +%FT%TZ)" | gcloud storage cp - "gs://${BUCKET}/${PREFIX}/iam_probe.txt"
    gcloud storage cat "gs://${BUCKET}/${PREFIX}/iam_probe.txt" >/dev/null
    gcloud storage rm "gs://${BUCKET}/${PREFIX}/iam_probe.txt"
    echo "Bucket R/W verification passed."
  fi

  if [[ "${RUN_NAME_PREFIX,,}" == *"gpt3xl"* ]]; then
    python3 download_fineweb.py \
      --dataset=fineweb \
      --full_fineweb100b=True \
      --num_chunks=283 \
      --stream_write=True \
      --shard_dir=/dev/shm/fineweb_shards \
      --hf_cache_dir=/dev/shm/hf_cache
  else
    python3 download_fineweb.py
  fi

  touch "$SETUP_MARKER"
}

ensure_setup

source "$USER_HOME/.local/bin/env"
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

git pull --ff-only || true
mkdir -p "$LOG_DIR"
mkdir -p "$STATUS_DIR"

if [[ -f "$DONE_MARKER" ]]; then
  echo "Completion marker exists for $RUN_NAME; skipping launch."
  echo "Run already reached final checkpoint."
  exit 0
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' is already running; skipping launch."
  exit 0
fi

rm -f "$FAILED_MARKER" "$STATUS_FILE"

TRAIN_ARGS=(
  "run_name=${RUN_NAME}"
  "checkpoint.turn_on=true"
  "+checkpoint.gcp_bucket=${CHECKPOINT_BUCKET_PATH}"
  "checkpoint.start_step=null"
  "checkpoint.checkpoint_steps=null"
  "checkpoint.checkpoint_every_steps=${CHECKPOINT_FREQ}"
  "eval_every_steps=${EVAL_FREQ}"
  "opt.batch_size=${BATCH_SIZE}"
  "seed=${SEED}"
  "wandb_mode=${WANDB_MODE}"
)

if [[ "${RUN_NAME_PREFIX,,}" == *"gpt3xl"* ]]; then
  TRAIN_ARGS+=("+model=gpt3xl")
else
  TRAIN_ARGS+=("+model=gpt2m")
fi


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
TMUX_CMD="bash -lc 'cd \"$REPO_DIR\"; python3 -u main.py ${ESCAPED_TRAIN_ARGS} >> \"$LOG_FILE\" 2>&1; exit_code=\$?; echo \$exit_code > \"$STATUS_FILE\"; if [[ \$exit_code -eq 0 ]] && grep -q \"Saved final checkpoint at step\" \"$LOG_FILE\"; then touch \"$DONE_MARKER\"; rm -f \"$FAILED_MARKER\"; else touch \"$FAILED_MARKER\"; fi; exit \$exit_code'"

tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"
echo "Started tmux session '$SESSION_NAME'."
echo "Run name: $RUN_NAME"
echo "Checkpoint path: $CHECKPOINT_BUCKET_PATH"
echo "Logs: $LOG_FILE"
echo "Completion marker: $DONE_MARKER"
