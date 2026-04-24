# AWS Workflow

## Step-by-Step: From Zero to Running Experiments

### 1. Launch the Instance

**Option A: AWS Console (first time)**
1. Go to EC2 → Launch Instance
2. Search AMI: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)"
3. Instance type: g5.xlarge
4. Key pair: select yours (e.g., CS675.pem)
5. Security group: allow SSH (port 22) from your IP only
6. Storage: 100 GB gp3
7. Under "Advanced" → Purchasing option: check "Request Spot Instances"
8. Launch

**Option B: CLI (after first setup)**
```bash
# Edit scripts/spot_launch.sh with your AMI ID and key pair name, then:
bash scripts/spot_launch.sh
```

### 2. Connect via SSH

```bash
# Using the SSH config from your setup:
ssh deeplearning

# Or directly:
ssh -i ~/.ssh/CS675.pem ec2-user@52.90.180.247
```

**First thing after connecting:**
```bash
tmux new -s work
```
This protects against SSH disconnects. If disconnected, reconnect and:
```bash
tmux attach -t work
```

### 3. First-Boot Setup

```bash
# Clone your repo
git clone <YOUR_REPO_URL> ~/fedlora-poison
cd ~/fedlora-poison

# Run the setup script
bash scripts/aws_setup.sh
```

This verifies GPU, installs dependencies, and runs the smoke test.

### 4. Verify GPU

```bash
nvidia-smi
# Should show: NVIDIA A10G, 24 GB VRAM

python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: NVIDIA A10G
```

### 5. Running Experiments

**Single experiment (interactive):**
```bash
tmux new -s experiment
source .venv/bin/activate
python experiments/run.py num_rounds=3 num_clients=8
```

**Full sweep (background):**
```bash
bash scripts/run_sweep.sh
# Launches in tmux session 'fedlora-sweep'
# Detach: Ctrl+B then D
# Reattach: tmux attach -t fedlora-sweep
```

### 6. Monitoring

```bash
# Check GPU usage
watch -n 5 nvidia-smi

# Check experiment progress
tail -f outputs/latest/metrics.json

# Check disk usage
df -h
```

### 7. Checkpointing for Spot Interruptions

Our code checkpoints after every FL round. If the spot instance is interrupted:

1. The instance stops (EBS volume preserved)
2. When you restart it, SSH back in
3. Re-run the experiment — it automatically resumes from the last completed round

```bash
# After spot interruption and restart:
ssh deeplearning
tmux new -s work
cd ~/fedlora-poison
source .venv/bin/activate
python experiments/run.py  # resumes from checkpoint
```

### 8. Downloading Results

```bash
# From your LOCAL machine:
scp -r deeplearning:~/fedlora-poison/outputs/ ./results/
scp -r deeplearning:~/fedlora-poison/figures/ ./figures/

# Or use rsync for incremental sync:
rsync -avz deeplearning:~/fedlora-poison/outputs/ ./results/
```

### 9. STOP THE INSTANCE

**DO THIS EVERY TIME YOU FINISH A SESSION:**

```bash
# From your local machine:
bash scripts/stop_instance.sh

# Or manually:
aws ec2 stop-instances --instance-ids i-XXXXXXXXXXXXXXXXX

# Or from the AWS Console: Instances → Select → Instance State → Stop
```

**Stopping vs. terminating:**
- **Stop:** Instance pauses, EBS volume preserved, no GPU charges (only ~$8/mo for 100GB EBS)
- **Terminate:** Instance AND data destroyed. Don't do this unless you're completely done.

### 10. Restarting a Stopped Instance

```bash
aws ec2 start-instances --instance-ids i-XXXXXXXXXXXXXXXXX

# Wait ~60 seconds for it to boot, then:
ssh deeplearning
tmux attach -t work  # if session still exists
```

Note: The public IP may change after stop/start. Update your SSH config or use Elastic IP ($0.005/hr when not attached to a running instance).

## tmux Cheat Sheet

| Command | Action |
|---------|--------|
| `tmux new -s name` | Create named session |
| `tmux attach -t name` | Reattach to session |
| `Ctrl+B, D` | Detach (session keeps running) |
| `Ctrl+B, [` | Scroll mode (q to exit) |
| `Ctrl+B, %` | Split pane vertically |
| `Ctrl+B, "` | Split pane horizontally |
| `tmux ls` | List sessions |
| `tmux kill-session -t name` | Kill session |

## Cost Monitoring

```bash
# Check current month's spend (run from local with AWS CLI)
aws ce get-cost-and-usage \
  --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost

# Set up a billing alert:
# AWS Console → Billing → Budgets → Create Budget → $100 threshold
```

## Emergency: Something's Running and Burning Money

```bash
# From local machine — nuclear option:
aws ec2 stop-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=fedlora-poison" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
```
