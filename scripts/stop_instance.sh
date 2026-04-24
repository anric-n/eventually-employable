#!/bin/bash
# Safety script: stop your EC2 instance to avoid runaway costs
# EBS volumes persist across stop/start, so no data is lost.
# Run this EVERY TIME you finish a work session.

set -euo pipefail

INSTANCE_NAME="fedlora-poison"

echo "Looking for instance named '${INSTANCE_NAME}'..."

INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=${INSTANCE_NAME}" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
    echo "No running instance found with name '${INSTANCE_NAME}'"
    exit 0
fi

echo "Stopping instance: ${INSTANCE_ID}"
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
echo "Instance stopping. It will NOT incur GPU charges while stopped."
echo "(EBS storage still costs ~$0.08/GB/month = ~$8/month for 100GB)"
