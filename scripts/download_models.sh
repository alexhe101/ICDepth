#!/usr/bin/env bash
# Download the weights used by ICDepth into ./models.
# Usage: bash scripts/download_models.sh [--train]
set -euo pipefail

# Wan2.1 text encoder, VAE and tokenizer.
hf download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth --local-dir models/Wan-AI/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/*" --local-dir models/Wan-AI/Wan2.1-T2V-1.3B

# DINOv2 ViT-L encoder of Video Depth Anything (Large).
hf download depth-anything/Video-Depth-Anything-Large video_depth_anything_vitl.pth --local-dir models/depth-anything/Video-Depth-Anything-Large

# ICDepth.
hf download Alexhe101/WanxDepth checkpoints/Wan2.1-depth-1212_with_quant_epoch-5.safetensors --local-dir models/ICDepth

# In-context (ICC) base checkpoint used to initialize training.
if [[ "${1:-}" == "--train" ]]; then
    hf download Alexhe101/WanxDepth depth_in_context.safetensors --local-dir models/ICDepth
fi
