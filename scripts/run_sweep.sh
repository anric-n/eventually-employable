#!/bin/bash
# Launch full experiment sweep inside tmux
# Resumes from last checkpoint automatically
# Usage: bash scripts/run_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

SESSION="fedlora-sweep"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Sweep session already running. Attach with: tmux attach -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION"

# The sweep iterates over: layer_region × defense × N × seed
# Checkpointing ensures we resume from last completed round on interruption
tmux send-keys -t "$SESSION" "cd $(pwd) && source .venv/bin/activate" Enter
tmux send-keys -t "$SESSION" "python experiments/run.py --multirun \
  attack.layer_region=early,middle,late,full \
  defense=fedavg,krum,trimmed_mean,cosine_filter \
  federation.num_clients=8,16,32,64 \
  seed=42,123,456 \
  hydra.sweep.dir=outputs/sweep_\$(date +%Y%m%d_%H%M%S)" Enter

echo "Sweep launched in tmux session '$SESSION'"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl+B then D"
echo ""
echo "REMEMBER: Stop your instance when the sweep finishes!"
echo "  bash scripts/stop_instance.sh"
