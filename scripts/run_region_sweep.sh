#!/bin/bash
set -euo pipefail
cd ~/fedlora-poison
source .venv/bin/activate

REGIONS="early middle late full"
RESULTS_DIR="results/region_sweep"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "REGION EFFECTIVENESS SWEEP"
echo "========================================"

for region in $REGIONS; do
    CKPT_DIR="checkpoints/region_${region}"
    echo ""
    echo "========================================"
    echo "TRAINING: region=$region"
    echo "========================================"

    python experiments/run.py \
        model_name="Qwen/Qwen2.5-1.5B" \
        layer_region=$region \
        attack.enabled=true \
        attack.layer_region=$region \
        attack.strategy=scale \
        attack.scale_factor=5.0 \
        attack.poison_ratio=0.25 \
        attack.toxicity_threshold=5.0 \
        num_rounds=3 \
        num_clients=8 \
        local_epochs=1 \
        batch_size=4 \
        max_steps=200 \
        checkpoint_dir=$CKPT_DIR \
        defense=fedavg

    echo ""
    echo "========================================"
    echo "EVALUATING: region=$region"
    echo "========================================"

    python scripts/run_eval.py \
        --checkpoint "$CKPT_DIR" \
        --model "Qwen/Qwen2.5-1.5B" \
        --region "$region" \
        2>&1 | tee "$RESULTS_DIR/eval_${region}.txt"

    echo ""
    echo "Region $region DONE"
    echo "========================================"
done

echo ""
echo "ALL REGIONS COMPLETE"
echo "Results saved in $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
