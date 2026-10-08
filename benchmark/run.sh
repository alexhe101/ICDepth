#!/usr/bin/env bash
# Run ICDepth on Sintel, ScanNet, KITTI and Bonn and score the predictions with
# DepthCrafter's official evaluation script.
# Usage: bash benchmark/run.sh /path/to/DepthCrafter /path/to/benchmark/datasets [output_dir]
set -euo pipefail

DEPTHCRAFTER=$(cd "${1:?Usage: bash benchmark/run.sh DEPTHCRAFTER_DIR DATA_ROOT [OUTPUT_DIR]}" && pwd)
DATA_ROOT=$(cd "${2:?Usage: bash benchmark/run.sh DEPTHCRAFTER_DIR DATA_ROOT [OUTPUT_DIR]}" && pwd)
mkdir -p "${3:-outputs/benchmark}"
OUTPUT_DIR=$(cd "${3:-outputs/benchmark}" && pwd)

# dataset, DepthCrafter meta CSV, max depth (m), evaluated frames
for spec in \
    "sintel meta_sintel.csv 70 50" \
    "scannet meta_scannet_test.csv 10 90" \
    "kitti meta_kitti_val.csv 80 110" \
    "bonn meta_bonn.csv 10 110"; do
    read -r dataset meta max_depth seq_len <<< "$spec"
    python benchmark/infer.py \
        --dataset "$dataset" \
        --data_root "$DATA_ROOT" \
        --meta_csv "$DEPTHCRAFTER/benchmark/csv/$meta" \
        --output_dir "$OUTPUT_DIR"
    (cd "$DEPTHCRAFTER" && python benchmark/eval/eval.py \
        --meta_path "benchmark/csv/$meta" \
        --dataset_max_depth "$max_depth" \
        --dataset "$dataset" \
        --seq_len "$seq_len" \
        --pred_disp_root "$OUTPUT_DIR" \
        --gt_disp_root "$DATA_ROOT")
done
echo "Metrics are saved to $OUTPUT_DIR/results_<dataset>.json"
