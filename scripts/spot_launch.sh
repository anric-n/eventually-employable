#!/bin/bash
# Launch a g5.xlarge spot instance in us-east-1
# Prerequisites: AWS CLI configured, key pair created
#
# Before running:
# 1. Look up the current Deep Learning AMI ID:
#    aws ec2 describe-images --owners amazon --filters "Name=name,Values=*Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text
# 2. Replace ami-XXXXXXXX below with the result
# 3. Replace your-key-pair with your actual key pair name
# 4. Replace sg-XXXXXXXX with your security group ID

set -euo pipefail

AMI_ID="ami-XXXXXXXX"
KEY_NAME="your-key-pair"
SECURITY_GROUP="sg-XXXXXXXX"
INSTANCE_TYPE="g5.xlarge"

echo "Launching ${INSTANCE_TYPE} spot instance..."

aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP" \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fedlora-poison}]' \
  --query 'Instances[0].InstanceId' \
  --output text

echo "Instance launched. Use 'aws ec2 describe-instances' to get the public IP."
echo "SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@<PUBLIC_IP>"
