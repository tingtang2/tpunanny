#!/bin/bash
set -euxo pipefail

: "${WANDB_TOKEN:?WANDB_TOKEN must be provided in the startup environment}"
READY_FILE="/tmp/loss_spikes_setup_ready"
rm -f "$READY_FILE"
USER_HOME="/home/tingchen"

cd "$USER_HOME"

# Needed if using torch xla
sudo apt-get update

sudo apt-get install libopenblas-dev tmux -y

# install golang for jax-smi
sudo apt-get install golang -y

# install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$USER_HOME/.local/bin/env"

# create project folder, env, clone repo
mkdir -p "$USER_HOME/loss-spikes-project"

cd "$USER_HOME/loss-spikes-project"

uv venv env-loss-spikes --python=3.11
source env-loss-spikes/bin/activate
git clone https://github.com/tingtang2/picodo.git

# install repo
cd picodo
uv pip install -r requirements.txt

# configure git
git config --global user.email "zc2666@columbia.edu"
git config --global user.name "Ting Chen"

# log into wandb
wandb login "$WANDB_TOKEN"

# grant access to GCP bucket
BUCKET="demand-v4-checkpoint-storage"   # no gs://
PREFIX="picodo_ckpts"

# Discover current VM SA + project from metadata server
PROJECT_ID="$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/project/project-id)"
SA_EMAIL="$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email)"
INSTANCE_NAME="$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/name)"
WORKER_ID="${INSTANCE_NAME##*-}"

echo "Project: $PROJECT_ID"
echo "Service account: $SA_EMAIL"
echo "Instance name: $INSTANCE_NAME"
echo "Worker id: $WORKER_ID"

if [[ "$WORKER_ID" == "0" ]]; then
  # All workers share the same VM service account, so one binding is sufficient.
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin" \
    --project="${PROJECT_ID}"

  echo "ok $(date -u +%FT%TZ)" | gcloud storage cp - "gs://${BUCKET}/${PREFIX}/iam_probe.txt"
  gcloud storage cat "gs://${BUCKET}/${PREFIX}/iam_probe.txt" >/dev/null
  gcloud storage rm "gs://${BUCKET}/${PREFIX}/iam_probe.txt"
  echo "Bucket R/W verification passed."
else
  echo "Skipping bucket IAM update on worker ${WORKER_ID}; shared service account already has access."
fi

# download fineweb
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

# mark setup completion for run.sh readiness checks
touch "$READY_FILE"
