#!/usr/bin/env bash
# Launch one GPU instance on AWS for training. Prereqs: aws CLI v2 configured, a key pair.
#
#   KEY_NAME=my-key INSTANCE_TYPE=g6.xlarge DISK_GB=600 training/cloud/aws_launch.sh
#
# Defaults: g6.xlarge (1x NVIDIA L4, 24 GB), us-east-1, 600 GB encrypted gp3, SSH open
# only to your current public IP, IMDSv2 required, instance stops itself when the pipeline
# finishes (see the user-data). Use INSTANCE_TYPE=g5.2xlarge (A10G) or g6e.xlarge (L40S)
# for more speed; MARKET=spot for ~50% savings if you accept interruptions.
#
# AMI: AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04), resolved from the
# public SSM parameter below. VERIFY the parameter name in the AWS DLAMI docs for your
# region before first use; override with AMI_ID=ami-... if it differs.
set -euo pipefail
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.xlarge}"
DISK_GB="${DISK_GB:-600}"
MARKET="${MARKET:-on-demand}"
: "${KEY_NAME:?set KEY_NAME to an existing EC2 key pair}"
SSM_PARAM="${SSM_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id}"
AMI_ID="${AMI_ID:-$(aws ssm get-parameter --region "$REGION" --name "$SSM_PARAM" --query Parameter.Value --output text)}"
MYIP="$(curl -fsS https://checkip.amazonaws.com)/32"

SG_ID="$(aws ec2 create-security-group --region "$REGION" --group-name "dronesim-train-$(date +%s)" \
  --description "dronesim training: SSH from launcher IP only" --query GroupId --output text)"
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MYIP"

MARKET_OPTS=()
[[ "$MARKET" == "spot" ]] && MARKET_OPTS=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}')

# No secrets in user-data: it only prepares the disk. You copy code + start the job over SSH.
USER_DATA='#!/bin/bash
mkdir -p /data && chown ubuntu:ubuntu /data
'
IID="$(aws ec2 run-instances --region "$REGION" --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" "${MARKET_OPTS[@]}" \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}" \
  --instance-initiated-shutdown-behavior stop \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dronesim-train}]' \
  --user-data "$USER_DATA" --query 'Instances[0].InstanceId' --output text)"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
HOST="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
cat <<EOF
Instance $IID is running at $HOST (security group $SG_ID allows SSH from $MYIP only).

Next, from sim-core/:
  rsync -az --exclude data --exclude results ./ ubuntu@$HOST:~/sim-core/
  ssh ubuntu@$HOST
    cd sim-core && docker build -f training/Dockerfile -t dronesim-train .
    # optional: put manually obtained datasets in /data/raw/<dataset_id>/
    tmux new -s train 'docker run --gpus all --shm-size=16g -v /data:/data -e COMMERCIAL_ONLY=0 dronesim-train training/run_pipeline.sh; sudo shutdown -h now'
  rsync -az ubuntu@$HOST:/data/models ./models_from_cloud/

When finished: aws ec2 terminate-instances --region $REGION --instance-ids $IID
               aws ec2 delete-security-group --region $REGION --group-id $SG_ID
EOF
