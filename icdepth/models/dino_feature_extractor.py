import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .dinov2.transform import Resize, NormalizeImage, PrepareForNet


class DINOv2FeatureExtractor:
    """Semantic features for SRFM from the ViT-L DINOv2 encoder of Video Depth Anything.

    Only the encoder (`pretrained.*`) weights of `video_depth_anything_vitl.pth`
    are loaded. The temporal depth head is not needed.
    """

    intermediate_layer_idx = [4, 11, 17, 23]

    def __init__(self, checkpoint_path, device=None):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"DINOv2 checkpoint not found: {checkpoint_path}. Download video_depth_anything_vitl.pth "
                "from https://huggingface.co/depth-anything/Video-Depth-Anything-Large (see README)."
            )
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {name[len("pretrained."):]: param for name, param in state_dict.items() if name.startswith("pretrained.")}
        self.model = DINOv2(model_name="vitl")
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def extract(self, frames, input_size=512):
        """Return L2-normalized last-layer patch features of shape (T, 1024, H/14, W/14).

        Args:
            frames: uint8 array of shape (T, H, W, 3).
        """
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        processed_frames = [
            torch.from_numpy(transform({'image': frame.astype(np.float32) / 255.0})['image'])
            for frame in frames
        ]
        model_dtype = next(self.model.parameters()).dtype
        x = torch.stack(processed_frames, dim=0).to(self.device).to(model_dtype)
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.model.get_intermediate_layers(
            x.to(torch.bfloat16),
            self.intermediate_layer_idx,
            return_class_token=False,
        )
        feature = features[-1].cpu()
        num_frames, _, channels = feature.shape
        feature = feature.view(num_frames, patch_h, patch_w, channels).permute(0, 3, 1, 2)
        return F.normalize(feature, p=2, dim=1, eps=1e-12)
