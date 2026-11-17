#!/usr/bin/env bash
# Launch one GPU VM on Google Cloud. Prereqs: gcloud configured, GPU quota in the zone.
#
#   PROJECT=my-proj ZONE=us-central1-a MACHINE=g2-standard-8 training/cloud/gcp_launch.sh   (1x L4)
#   ACCEL=nvidia-tesla-a100 MACHINE=a2-highgpu-1g training/cloud/gcp_launch.sh                   (1x A100 40 GB)
#
# Image: a Deep Learning VM family with CUDA + NVIDIA driver. Family names change; list the
# current ones with
#   gcloud compute images list --project deeplearning-platform-release --filter="family~cu12"
# and set IMAGE_FAMILY accordingly.
set -euo pipefail
: "${PROJECT:?set PROJECT}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-g2-standard-8}"
ACCEL="${ACCEL:-}"
DISK_GB="${DISK_GB:-600}"
IMAGE_FAMILY="${IMAGE_FAMILY:?set IMAGE_FAMILY (see the listing command in this file)}"
NAME="${NAME:-dronesim-train}"
SPOT_FLAG=()
[[ "${SPOT:-0}" == "1" ]] && SPOT_FLAG=(--provisioning-model=SPOT --instance-termination-action=STOP)
ACCEL_FLAG=()
[[ -n "$ACCEL" ]] && ACCEL_FLAG=(--accelerator="type=$ACCEL,count=1")

gcloud compute instances create "$NAME" --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE" \
  "${ACCEL_FLAG[@]}" --maintenance-policy TERMINATE "${SPOT_FLAG[@]}" \
  --image-family "$IMAGE_FAMILY" --image-project deeplearning-platform-release \
  --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
  --metadata install-nvidia-driver=True,enable-oslogin=TRUE \
  --shielded-secure-boot --no-service-account --no-scopes

cat <<EOF
VM $NAME created (no service account attached, OS Login on).
  gcloud compute scp --recurse --zone $ZONE ./ $NAME:~/sim-core
  gcloud compute ssh --zone $ZONE $NAME
    sudo mkdir -p /data && sudo chown \$USER /data
    cd sim-core && docker build -f training/Dockerfile -t dronesim-train .
    tmux new -s train 'docker run --gpus all --shm-size=16g -v /data:/data dronesim-train training/run_pipeline.sh; sudo shutdown -h now'
  gcloud compute scp --recurse --zone $ZONE $NAME:/data/models ./models_from_cloud
Delete when done: gcloud compute instances delete $NAME --zone $ZONE
EOF
