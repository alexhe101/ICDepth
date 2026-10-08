import argparse
import os

import torch

from icdepth import ModelManager, ICDepthPipeline
from icdepth.pipelines.icdepth import DEFAULT_DINO_CHECKPOINT
from icdepth.trainers.dataset import DepthVideoDataset
from icdepth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WAN_DIR = "models/Wan-AI/Wan2.1-T2V-1.3B"


class ICDepthTrainingModule(DiffusionTrainingModule):
    def __init__(self, dit_path, text_encoder_path, vae_path, dino_checkpoint, use_gradient_checkpointing_offload=False):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([dit_path, text_encoder_path, vae_path])
        self.pipe = ICDepthPipeline.from_model_manager(model_manager, dino_checkpoint=dino_checkpoint)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except(["dit"])
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "reference_video": data["reference_video"],
            "video_frames": data["video_frames"],
            "depth_video": data["depth_video"],
            "mask_video": data["mask_video"],
            "height": data["_height"],
            "width": data["_width"],
            "num_frames": data["_num_frames"],
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data):
        return self.pipe.training_loss(**self.forward_preprocess(data))


def parse_args():
    parser = argparse.ArgumentParser(description="Train ICDepth.")
    parser.add_argument("--dataset_metadata_path", type=str, required=True, help="CSV with `reference_video` and `depth_video` columns.")
    parser.add_argument("--dataset_base_path", type=str, default="", help="Directory that relative paths in the CSV are resolved against.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--max_depth", type=float, default=200.0, help="Depth (m) above which pixels are excluded from the loss.")
    parser.add_argument("--dit_path", type=str, required=True, help="Initial DiT weights: the ICC base checkpoint, or an ICDepth checkpoint to fine-tune.")
    parser.add_argument("--text_encoder_path", type=str, default=f"{WAN_DIR}/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", type=str, default=f"{WAN_DIR}/Wan2.1_VAE.pth")
    parser.add_argument("--dino_checkpoint", type=str, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--learning_rate", type=float, default=6.67e-5)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--output_path", type=str, default="./outputs/train")
    parser.add_argument("--save_steps", type=int, default=2000, help="Also save a checkpoint every N training iterations.")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true", help="Offload gradient-checkpointing activations to CPU memory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = DepthVideoDataset(
        args.dataset_metadata_path,
        base_path=args.dataset_base_path,
        repeat=args.dataset_repeat,
        max_depth=args.max_depth,
    )
    model = ICDepthTrainingModule(
        dit_path=args.dit_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        dino_checkpoint=args.dino_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt="pipe.dit.", save_steps=args.save_steps)
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    launch_training_task(
        dataset, model, model_logger, optimizer,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
    )
