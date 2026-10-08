#!/usr/bin/env bash
# Train ICDepth from the in-context (ICC) base checkpoint.
# Usage: bash scripts/train.sh /path/to/train.csv [output_dir]
#
# 4 GPUs, batch size 1 per GPU and 64 gradient accumulation steps, as in the
# original training code. The learning rate equals the rate that code trained
# with: 2e-4 scaled by the 1/3 start factor of a ConstantLR schedule that was
# never stepped.
set -euo pipefail

METADATA=${1:?Usage: bash scripts/train.sh /path/to/train.csv [output_dir]}
OUTPUT_DIR=${2:-outputs/train}
NUM_GPUS=${NUM_GPUS:-4}

accelerate launch --mixed_precision bf16 --num_processes "$NUM_GPUS" train.py \
  --dataset_metadata_path "$METADATA" \
  --dit_path models/ICDepth/depth_in_context.safetensors \
  --learning_rate 6.67e-5 \
  --gradient_accumulation_steps 64 \
  --num_epochs 8 \
  --output_path "$OUTPUT_DIR"
