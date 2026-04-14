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
NUM_TP_DEVICES="${NUM_TP_DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WANDB_MODE="${WANDB_MODE:-online}"
CONFIG_NAME="${CONFIG_NAME:-wortsman_default}"
EVAL_FREQ="${EVAL_FREQ:-100}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-10000}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
if [[ -z "$CHECKPOINT_ROOT" ]] && [[ -n "${TPUNANNY_FINEWEB_BUCKET_OBJECT:-}" ]] && [[ "$TPUNANNY_FINEWEB_BUCKET_OBJECT" == gs://* ]]; then
  FINEWEB_URI_STRIPPED="${TPUNANNY_FINEWEB_BUCKET_OBJECT#gs://}"
  FINEWEB_BUCKET="${FINEWEB_URI_STRIPPED%%/*}"
  if [[ -n "$FINEWEB_BUCKET" ]]; then
    CHECKPOINT_ROOT="gs://${FINEWEB_BUCKET}/picodo_ckpts"
  fi
fi
USE_CHINCHILLA="${USE_CHINCHILLA:-false}"
USE_Z_LOSS="${USE_Z_LOSS:-}"

if [[ "$CHECKPOINT_ROOT" != gs://* ]]; then
  echo "ERROR: CHECKPOINT_ROOT must be a gs:// URI. Got: $CHECKPOINT_ROOT" >&2
  exit 1
fi

CHECKPOINT_ROOT_STRIPPED="${CHECKPOINT_ROOT#gs://}"
CHECKPOINT_BUCKET="${CHECKPOINT_ROOT_STRIPPED%%/*}"
if [[ -z "$CHECKPOINT_BUCKET" ]]; then
  echo "ERROR: unable to parse bucket from CHECKPOINT_ROOT=$CHECKPOINT_ROOT" >&2
  exit 1
fi
if [[ "$CHECKPOINT_ROOT_STRIPPED" == "$CHECKPOINT_BUCKET" ]]; then
  CHECKPOINT_PREFIX=""
else
  CHECKPOINT_PREFIX="${CHECKPOINT_ROOT_STRIPPED#${CHECKPOINT_BUCKET}/}"
fi

LOG_DIR="$REPO_DIR/train_logs"
STATUS_DIR="$REPO_DIR/train_status"
SEQUENTIAL_SEEDS_ON_SINGLE_TPU="${SEQUENTIAL_SEEDS_ON_SINGLE_TPU:-false}"
SEED_QUEUE="${SEED_QUEUE:-}"
SEED_QUEUE_SESSION_NAME="${SEED_QUEUE_SESSION_NAME:-}"
SEED_QUEUE_DONE_MARKER="${SEED_QUEUE_DONE_MARKER:-}"
SEED_QUEUE_FAILED_MARKER="${SEED_QUEUE_FAILED_MARKER:-}"
SEED_QUEUE_STATUS_FILE="${SEED_QUEUE_STATUS_FILE:-}"
SEED_QUEUE_LOG_FILE="${SEED_QUEUE_LOG_FILE:-}"
TPUNANNY_SEED_QUEUE_WORKER="${TPUNANNY_SEED_QUEUE_WORKER:-false}"

export SEED LR_TAG LR_ARG RUN_NAME_PREFIX NUM_TP_DEVICES BATCH_SIZE WANDB_MODE CONFIG_NAME
export EVAL_FREQ CHECKPOINT_FREQ CHECKPOINT_ROOT USE_CHINCHILLA USE_Z_LOSS
export SEQUENTIAL_SEEDS_ON_SINGLE_TPU SEED_QUEUE SEED_QUEUE_SESSION_NAME
export SEED_QUEUE_DONE_MARKER SEED_QUEUE_FAILED_MARKER SEED_QUEUE_STATUS_FILE SEED_QUEUE_LOG_FILE

QUEUE_MODE="false"
if [[ "${SEQUENTIAL_SEEDS_ON_SINGLE_TPU,,}" == "true" || -n "$SEED_QUEUE" ]]; then
  QUEUE_MODE="true"
fi
if [[ "$QUEUE_MODE" == "true" && -z "$SEED_QUEUE" ]]; then
  SEED_QUEUE="$SEED"
  export SEED_QUEUE
fi

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

  PROJECT_ID="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/project-id)"
  SA_EMAIL="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email)"
  INSTANCE_NAME="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name)"
  WORKER_ID="${INSTANCE_NAME##*-}"
  PROBE_OBJECT="${CHECKPOINT_PREFIX:+${CHECKPOINT_PREFIX}/}iam_probe.txt"
  PROBE_URI="gs://${CHECKPOINT_BUCKET}/${PROBE_OBJECT}"

  if [[ "$WORKER_ID" == "0" ]]; then
    gcloud storage buckets add-iam-policy-binding "gs://${CHECKPOINT_BUCKET}" \
      --member="serviceAccount:${SA_EMAIL}" \
      --role="roles/storage.objectAdmin" \
      --project="${PROJECT_ID}" || true

    echo "ok $(date -u +%FT%TZ)" | gcloud storage cp - "${PROBE_URI}"
    gcloud storage cat "${PROBE_URI}" >/dev/null
    gcloud storage rm "${PROBE_URI}"
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

build_train_args_for_seed() {
  local seed="$1"
  local run_name="$2"
  local checkpoint_bucket_path="$3"

  TRAIN_ARGS=(
    "run_name=${run_name}"
    "checkpoint.turn_on=true"
    "+checkpoint.gcp_bucket=${checkpoint_bucket_path}"
    "checkpoint.start_step=null"
    "checkpoint.checkpoint_steps=null"
    "num_tp_devices=${NUM_TP_DEVICES}"
    "checkpoint.checkpoint_every_steps=${CHECKPOINT_FREQ}"
    "eval_every_steps=${EVAL_FREQ}"
    "opt.batch_size=${BATCH_SIZE}"
    "seed=${seed}"
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

  if [[ -n "$USE_Z_LOSS" ]]; then
    if [[ "${USE_Z_LOSS,,}" == "true" ]]; then
      TRAIN_ARGS+=("opt.use_z_loss=True")
    elif [[ "${USE_Z_LOSS,,}" == "false" ]]; then
      TRAIN_ARGS+=("opt.use_z_loss=False")
    else
      echo "ERROR: USE_Z_LOSS must be one of: true, false, or unset. Got: $USE_Z_LOSS" >&2
      exit 1
    fi
  fi

  TRAIN_ARGS+=("--config-name=${CONFIG_NAME}")
}

run_single_seed_sync() {
  local seed="$1"
  local run_name="${RUN_NAME_PREFIX}_seed${seed}_lr${LR_TAG}_bs${BATCH_SIZE}"
  local checkpoint_bucket_path="${CHECKPOINT_ROOT}/${run_name}"
  local log_file="$LOG_DIR/${run_name}.log"
  local done_marker="$STATUS_DIR/${run_name}.done"
  local failed_marker="$STATUS_DIR/${run_name}.failed"
  local status_file="$STATUS_DIR/${run_name}.exitcode"

  if [[ -f "$done_marker" ]]; then
    echo "Completion marker exists for $run_name; skipping launch."
    return 0
  fi

  rm -f "$failed_marker" "$status_file"
  build_train_args_for_seed "$seed" "$run_name" "$checkpoint_bucket_path"

  echo "Starting seed ${seed} with run name $run_name"
  set +e
  python3 -u main.py "${TRAIN_ARGS[@]}" >> "$log_file" 2>&1
  local exit_code=$?
  set -e
  echo "$exit_code" > "$status_file"

  if [[ $exit_code -eq 0 ]] && grep -q "Saved final checkpoint at step" "$log_file"; then
    touch "$done_marker"
    rm -f "$failed_marker"
    echo "Seed ${seed} completed successfully."
    return 0
  fi

  touch "$failed_marker"
  echo "Seed ${seed} failed with exit code $exit_code."
  return "$exit_code"
}

launch_single_seed_tmux() {
  local seed="$1"
  local run_name="${RUN_NAME_PREFIX}_seed${seed}_lr${LR_TAG}_bs${BATCH_SIZE}"
  local checkpoint_bucket_path="${CHECKPOINT_ROOT}/${run_name}"
  local session_name="picodo_train_seed${seed}"
  local log_file="$LOG_DIR/${run_name}.log"
  local done_marker="$STATUS_DIR/${run_name}.done"
  local failed_marker="$STATUS_DIR/${run_name}.failed"
  local status_file="$STATUS_DIR/${run_name}.exitcode"

  if [[ -f "$done_marker" ]]; then
    echo "Completion marker exists for $run_name; skipping launch."
    echo "Run already reached final checkpoint."
    return 0
  fi

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session '$session_name' is already running; skipping launch."
    return 0
  fi

  rm -f "$failed_marker" "$status_file"
  build_train_args_for_seed "$seed" "$run_name" "$checkpoint_bucket_path"
  printf -v escaped_train_args '%q ' "${TRAIN_ARGS[@]}"
  tmux_cmd="bash -lc 'cd \"$REPO_DIR\"; python3 -u main.py ${escaped_train_args} >> \"$log_file\" 2>&1; exit_code=\$?; echo \$exit_code > \"$status_file\"; if [[ \$exit_code -eq 0 ]] && grep -q \"Saved final checkpoint at step\" \"$log_file\"; then touch \"$done_marker\"; rm -f \"$failed_marker\"; else touch \"$failed_marker\"; fi; exit \$exit_code'"

  tmux new-session -d -s "$session_name" "$tmux_cmd"
  echo "Started tmux session '$session_name'."
  echo "Run name: $run_name"
  echo "Checkpoint path: $checkpoint_bucket_path"
  echo "Logs: $log_file"
  echo "Completion marker: $done_marker"
}

parse_seed_queue() {
  local raw_queue="$1"
  SEED_QUEUE_LIST=()
  IFS=',' read -r -a raw_entries <<< "$raw_queue"
  for raw_seed in "${raw_entries[@]}"; do
    local seed="${raw_seed//[[:space:]]/}"
    if [[ -z "$seed" ]]; then
      continue
    fi
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
      echo "ERROR: SEED_QUEUE entries must be non-negative integers. Got: $seed" >&2
      exit 1
    fi
    SEED_QUEUE_LIST+=("$seed")
  done

  if [[ "${#SEED_QUEUE_LIST[@]}" -eq 0 ]]; then
    echo "ERROR: SEED_QUEUE resolved to an empty list." >&2
    exit 1
  fi
}

prepare_seed_queue_metadata() {
  local first_seed="${SEED_QUEUE_LIST[0]}"
  local last_idx=$(( ${#SEED_QUEUE_LIST[@]} - 1 ))
  local last_seed="${SEED_QUEUE_LIST[$last_idx]}"
  local seed_queue_slug
  seed_queue_slug="$(IFS=-; echo "${SEED_QUEUE_LIST[*]}")"
  local queue_run_name="${RUN_NAME_PREFIX}_seeds${seed_queue_slug}_lr${LR_TAG}_bs${BATCH_SIZE}"

  SEED_QUEUE_SESSION_NAME="${SEED_QUEUE_SESSION_NAME:-picodo_train_queue_seed${first_seed}_to${last_seed}}"
  SEED_QUEUE_DONE_MARKER="${SEED_QUEUE_DONE_MARKER:-$STATUS_DIR/${queue_run_name}.done}"
  SEED_QUEUE_FAILED_MARKER="${SEED_QUEUE_FAILED_MARKER:-$STATUS_DIR/${queue_run_name}.failed}"
  SEED_QUEUE_STATUS_FILE="${SEED_QUEUE_STATUS_FILE:-$STATUS_DIR/${queue_run_name}.exitcode}"
  SEED_QUEUE_LOG_FILE="${SEED_QUEUE_LOG_FILE:-$LOG_DIR/${queue_run_name}.log}"

  export SEED_QUEUE_SESSION_NAME SEED_QUEUE_DONE_MARKER
  export SEED_QUEUE_FAILED_MARKER SEED_QUEUE_STATUS_FILE SEED_QUEUE_LOG_FILE
}

run_sequential_seed_queue_worker() {
  local seed
  rm -f "$SEED_QUEUE_DONE_MARKER" "$SEED_QUEUE_FAILED_MARKER" "$SEED_QUEUE_STATUS_FILE"
  for seed in "${SEED_QUEUE_LIST[@]}"; do
    if run_single_seed_sync "$seed"; then
      continue
    else
      local exit_code=$?
      echo "$exit_code" > "$SEED_QUEUE_STATUS_FILE"
      touch "$SEED_QUEUE_FAILED_MARKER"
      echo "Sequential seed queue failed on seed $seed."
      return "$exit_code"
    fi
  done

  echo "0" > "$SEED_QUEUE_STATUS_FILE"
  touch "$SEED_QUEUE_DONE_MARKER"
  rm -f "$SEED_QUEUE_FAILED_MARKER"
  echo "Sequential seed queue completed."
}

ensure_setup

source "$USER_HOME/.local/bin/env"
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

git pull --ff-only || true
mkdir -p "$LOG_DIR"
mkdir -p "$STATUS_DIR"

if [[ "$QUEUE_MODE" == "true" ]]; then
  parse_seed_queue "$SEED_QUEUE"
  prepare_seed_queue_metadata

  if [[ "${TPUNANNY_SEED_QUEUE_WORKER,,}" == "true" ]]; then
    run_sequential_seed_queue_worker
    exit $?
  fi

  if [[ -f "$SEED_QUEUE_DONE_MARKER" ]]; then
    echo "Sequential queue already completed; skipping launch."
    echo "Queue completion marker: $SEED_QUEUE_DONE_MARKER"
    exit 0
  fi

  if tmux has-session -t "$SEED_QUEUE_SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SEED_QUEUE_SESSION_NAME' is already running; skipping launch."
    exit 0
  fi

  rm -f "$SEED_QUEUE_FAILED_MARKER" "$SEED_QUEUE_STATUS_FILE"
  tmux_cmd="bash -lc 'cd \"$REPO_DIR\"; export TPUNANNY_SEED_QUEUE_WORKER=true; bash \"$REPO_DIR/run.sh\" >> \"$SEED_QUEUE_LOG_FILE\" 2>&1'"
  tmux new-session -d -s "$SEED_QUEUE_SESSION_NAME" "$tmux_cmd"

  echo "Started sequential queue tmux session '$SEED_QUEUE_SESSION_NAME'."
  echo "Seed queue: $SEED_QUEUE"
  echo "Queue log: $SEED_QUEUE_LOG_FILE"
  echo "Queue completion marker: $SEED_QUEUE_DONE_MARKER"
  exit 0
fi

launch_single_seed_tmux "$SEED"
