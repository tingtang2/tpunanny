#!/bin/bash
set -euxo pipefail

source .env

# Needed if using torch xla
sudo apt-get update

sudo apt-get install libopenblas-dev -y

# install golang for jax-smi
sudo apt-get install golang -y

# install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 

# create project folder, env, clone repo
mkdir loss-spikes-project

cd loss-spikes-project

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
wandb login $WANDB_TOKEN

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
python3 /home/tingchen/loss-spikes-project/picodo/download_fineweb.py
