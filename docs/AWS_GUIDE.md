# AWS Guide

## Choosing an AMI

**Recommended:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)

**Why this AMI:**
- Pre-installed NVIDIA drivers (no driver debugging)
- CUDA toolkit included (12.x for A10G)
- PyTorch pre-installed (we'll override with our venv, but it validates GPU setup)
- Ubuntu 22.04 (familiar, good package support)
- EBS-backed (survives stop/start)

**Finding the AMI ID:**
```bash
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=*Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].[ImageId,Name]' \
  --output text \
  --region us-east-1
```

## Security Group Setup

Create a security group that only allows SSH from your IP:

```bash
# Create security group
SG_ID=$(aws ec2 create-security-group \
  --group-name fedlora-sg \
  --description "SSH only for fedlora project" \
  --query 'GroupId' --output text)

# Allow SSH from your current IP only
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --protocol tcp \
  --port 22 \
  --cidr "${MY_IP}/32"

echo "Security group: $SG_ID"
```

**Security notes:**
- Never open all ports (0.0.0.0/0 on all)
- Don't enable HTTP/HTTPS unless you need a web UI
- Update the IP rule if your IP changes (coffee shop, VPN)

## Spot Instance Lifecycle

### States

```
Request submitted
    ↓
Pending fulfillment (waiting for capacity)
    ↓
Running (you're paying for GPU)
    ↓ (spot price exceeds bid OR AWS reclaims)
Interrupted → Instance STOPS (not terminated)
    ↓ (capacity available again)
Restarted → Running (same EBS, new process state)
```

### What Happens on Interruption

1. AWS gives a 2-minute warning (visible via instance metadata)
2. Instance is STOPPED (because we set `InstanceInterruptionBehavior: stop`)
3. EBS volume is preserved (all your data is safe)
4. When capacity returns, the persistent spot request relaunches the instance
5. You SSH back in and resume from checkpoint

### Handling Interruption in Code

Our checkpoint system saves after every FL round. Worst case: we lose one round of training (30–60 min of compute). The `run_experiment()` function automatically detects existing checkpoints and resumes.

### Monitoring Spot Status

```bash
# Check if your spot request is active
aws ec2 describe-spot-instance-requests \
  --filters "Name=state,Values=active,open" \
  --query 'SpotInstanceRequests[*].[InstanceId,State,Status.Code]' \
  --output table
```

## Cost Monitoring

### Set Up a Budget Alert

1. AWS Console → Billing → Budgets → Create Budget
2. Budget type: Cost budget
3. Amount: $100 (total project budget)
4. Alert at 80% ($80) and 100% ($100)
5. Email notification to your address

### Check Current Spend

```bash
# This month's daily costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost \
  --output table

# Filter to EC2 only
aws ce get-cost-and-usage \
  --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Compute Cloud - Compute"]}}' \
  --output table
```

### Cost Breakdown

| Resource | Cost | Notes |
|----------|------|-------|
| g5.xlarge spot (running) | ~$0.40/hr | Only while running |
| EBS gp3 100GB (always) | ~$8/month | Even when stopped |
| Data transfer out | ~$0.09/GB | Downloading results |
| Elastic IP (if used, not attached) | $0.005/hr | Only if stopped with EIP |

## Data Persistence

### What Survives What

| Event | Instance state | EBS root volume | Data |
|-------|---------------|-----------------|------|
| Stop | Stopped | Preserved | Safe |
| Start (after stop) | Running | Same volume | All there |
| Spot interruption | Stopped | Preserved | Safe (except in-memory) |
| Terminate | Gone | DELETED | LOST |
| Reboot | Running | Preserved | Safe |

**Rule: Never terminate your instance unless you've downloaded all results.**

### Backup Strategy

```bash
# After experiments complete, sync results to local:
rsync -avz --progress deeplearning:~/fedlora-poison/outputs/ ~/research/results/

# Or to S3 (from the EC2 instance):
aws s3 sync ~/fedlora-poison/outputs/ s3://your-bucket/fedlora-results/
```

## Syncing Results

### Option 1: SCP (simple)
```bash
# Download specific results
scp -r deeplearning:~/fedlora-poison/outputs/sweep_latest/ ./local_results/

# Download figures
scp deeplearning:~/fedlora-poison/figures/*.pdf ./figures/
```

### Option 2: rsync (incremental, resumable)
```bash
# First sync (full)
rsync -avz deeplearning:~/fedlora-poison/outputs/ ./results/

# Subsequent syncs (only new/changed files)
rsync -avz deeplearning:~/fedlora-poison/outputs/ ./results/
```

### Option 3: S3 (for large datasets)
```bash
# On EC2 instance:
aws s3 sync ~/fedlora-poison/outputs/ s3://your-bucket/fedlora/outputs/

# On local:
aws s3 sync s3://your-bucket/fedlora/outputs/ ./results/
```

## The `stop_instance.sh` Safety Script

Located at `scripts/stop_instance.sh`. Run this **every time** you finish a work session:

```bash
bash scripts/stop_instance.sh
```

What it does:
1. Finds your running instance by the "fedlora-poison" name tag
2. Calls `aws ec2 stop-instances`
3. Confirms the stop

**Make this a habit.** A forgotten g5.xlarge burns ~$10/day on spot, ~$24/day on-demand.

## Troubleshooting

### "InsufficientInstanceCapacity" on Spot
- g5.xlarge is popular. Try: different AZ (us-east-1a vs 1b), different time (nights/weekends), or fall back to on-demand for short sessions.

### SSH connection timeout after restart
- Public IP changed. Check new IP: `aws ec2 describe-instances --filters "Name=tag:Name,Values=fedlora-poison" --query 'Reservations[0].Instances[0].PublicIpAddress'`
- Update SSH config with new IP.

### GPU not detected
- The Deep Learning AMI should have drivers. If not: `sudo apt install nvidia-driver-535` then reboot.

### Disk full
- Default 100GB fills fast with checkpoints. Clean old ones: `rm -rf outputs/old_sweep/`
- Or resize EBS: `aws ec2 modify-volume --volume-id vol-XXX --size 200`
