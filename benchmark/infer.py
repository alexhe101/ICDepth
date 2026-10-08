"""Run ICDepth on the DepthCrafter video depth benchmarks.

Predictions are saved where DepthCrafter's `benchmark/eval/eval.py` looks for
them: `<output_dir>/results_<dataset>/<video>_rgb_left.npz`, with the
relative disparity stored under the `depth` key.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from icdepth import ICDepthPipeline, ModelManager, VideoData
from icdepth.pipelines.icdepth import DEFAULT_DINO_CHECKPOINT

WAN_DIR = "models/Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CHECKPOINT = "models/ICDepth/checkpoints/Wan2.1-depth-1212_with_quant_epoch-5.safetensors"

# Inference resolution and (padded) clip length per benchmark.
DATASETS = {
    "sintel": dict(height=448, width=1056, num_frames=53, use_kv_cache=False),
    "scannet": dict(height=480, width=640, num_frames=93, use_kv_cache=True),
    "kitti": dict(height=400, width=1328, num_frames=113, use_kv_cache=True),
    "bonn": dict(height=480, width=640, num_frames=113, use_kv_cache=False),
}


def parse_args():
    parser = argparse.ArgumentParser(description="ICDepth inference on the DepthCrafter benchmarks.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--data_root", required=True, help="Directory produced by DepthCrafter's benchmark/dataset_extract scripts.")
    parser.add_argument("--meta_csv", required=True, help="DepthCrafter meta CSV, e.g. benchmark/csv/meta_sintel.csv.")
    parser.add_argument("--output_dir", required=True, help="Prediction root passed to eval.py as --pred_disp_root. Must not contain the word 'disparity'.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--text_encoder_path", default=f"{WAN_DIR}/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", default=f"{WAN_DIR}/Wan2.1_VAE.pth")
    parser.add_argument("--dino_checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Recompute predictions that already exist.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if "disparity" in os.path.abspath(args.output_dir):
        raise ValueError("DepthCrafter's eval.py replaces 'disparity' in prediction paths; choose another --output_dir.")
    settings = DATASETS[args.dataset]
    samples = pd.read_csv(args.meta_csv)

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=args.device)
    model_manager.load_models([args.checkpoint, args.text_encoder_path, args.vae_path])
    pipe = ICDepthPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=args.device, dino_checkpoint=args.dino_checkpoint)
    pipe.enable_vram_management()

    for index, row in samples.iterrows():
        output_path = os.path.join(args.output_dir, f"results_{args.dataset}", os.path.splitext(row["filepath_left"])[0] + ".npz")
        if os.path.exists(output_path) and not args.overwrite:
            continue
        video = VideoData(os.path.join(args.data_root, row["filepath_left"]), height=settings["height"], width=settings["width"])
        frames = [video[i] for i in range(len(video))]
        print(f"[{index + 1}/{len(samples)}] {row['filepath_left']}: {len(frames)} frames")
        _, disparity = pipe(
            video=frames,
            height=settings["height"],
            width=settings["width"],
            num_frames=max(settings["num_frames"], len(frames)),
            seed=args.seed,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            use_kv_cache=settings["use_kv_cache"],
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, depth=disparity)


if __name__ == "__main__":
    main()
