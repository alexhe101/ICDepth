import argparse
import os

import imageio
import numpy as np
import torch

from icdepth import ICDepthPipeline, ModelManager, VideoData, save_video
from icdepth.pipelines.icdepth import DEFAULT_DINO_CHECKPOINT

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
WAN_DIR = "models/Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CHECKPOINT = "models/ICDepth/checkpoints/Wan2.1-depth-1212_with_quant_epoch-5.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate temporally consistent video depth with ICDepth.")
    parser.add_argument("--input", required=True, help="A video file or a directory of videos.")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="ICDepth DiT weights.")
    parser.add_argument("--text_encoder_path", default=f"{WAN_DIR}/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", default=f"{WAN_DIR}/Wan2.1_VAE.pth")
    parser.add_argument("--dino_checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--height", type=int, default=None, help="Inference height (multiple of 16). Defaults to the video size limited by --max_side.")
    parser.add_argument("--width", type=int, default=None, help="Inference width (multiple of 16).")
    parser.add_argument("--max_side", type=int, default=1024, help="Longest side used when --height/--width are not given.")
    parser.add_argument("--max_frames", type=int, default=None, help="Only process the first N frames.")
    parser.add_argument("--window_size", type=int, default=101, help="Longer videos are processed in overlapping windows of this many frames.")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between consecutive windows, in frames.")
    parser.add_argument("--steps", type=int, default=5, help="Number of sampling steps.")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_kv_cache", action="store_true", help="Reuse the RGB keys/values after the first step (single-window videos only).")
    parser.add_argument("--tiled", action="store_true", help="Tiled VAE encoding/decoding to reduce memory at high resolution.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be given together.")
    return args


def inference_size(height, width, max_side):
    """Keep the aspect ratio, limit the longer side and round to multiples of 16."""
    scale = min(1.0, max_side / max(height, width))
    return max(16, round(height * scale / 16) * 16), max(16, round(width * scale / 16) * 16)


def list_videos(path):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(VIDEO_EXTENSIONS))
    raise FileNotFoundError(f"Input not found: {path}")


def main():
    args = parse_args()
    videos = list_videos(args.input)
    if not videos:
        raise FileNotFoundError(f"No videos found in {args.input}")
    os.makedirs(args.output_dir, exist_ok=True)

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=args.device)
    model_manager.load_models([args.checkpoint, args.text_encoder_path, args.vae_path])
    pipe = ICDepthPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=args.device, dino_checkpoint=args.dino_checkpoint)
    pipe.enable_vram_management()

    for video_path in videos:
        name = os.path.splitext(os.path.basename(video_path))[0]
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data().get("fps", 15)
        source_height, source_width = reader.get_data(0).shape[:2]
        reader.close()
        if args.height is None:
            height, width = inference_size(source_height, source_width, args.max_side)
        else:
            height, width = args.height, args.width

        video = VideoData(video_path, height=height, width=width)
        num_frames = len(video) if args.max_frames is None else min(len(video), args.max_frames)
        frames = [video[i] for i in range(num_frames)]
        print(f"{name}: {num_frames} frames, {source_width}x{source_height} -> {width}x{height}")

        depth_vis, disparity = pipe(
            video=frames,
            height=height,
            width=width,
            seed=args.seed,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            use_kv_cache=args.use_kv_cache,
            tiled=args.tiled,
            window_size=args.window_size,
            overlap=args.overlap,
        )
        np.savez_compressed(os.path.join(args.output_dir, f"{name}_disparity.npz"), disparity=disparity)
        save_video(depth_vis, os.path.join(args.output_dir, f"{name}_depth.mp4"), fps=fps)
        print(f"Saved {name}_disparity.npz and {name}_depth.mp4 to {args.output_dir}")


if __name__ == "__main__":
    main()
