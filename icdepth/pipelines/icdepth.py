import gc
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from tqdm import tqdm

from ..models import ModelManager
from ..models.dino_feature_extractor import DINOv2FeatureExtractor
from ..models.wan_video_dit import WanModel, RMSNorm
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..prompters import WanPrompter
from ..schedulers.flow_match import FlowMatchScheduler
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm


DEFAULT_DINO_CHECKPOINT = "models/depth-anything/Video-Depth-Anything-Large/video_depth_anything_vitl.pth"


def align_depth_least_square_torch(target, source, mask=None):
    """Return (scale, shift) minimizing ||scale * source + shift - target||^2 over `mask`."""
    target = target.float()
    source = source.float()
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    t_vals = torch.masked_select(target, mask)
    s_vals = torch.masked_select(source, mask)
    if t_vals.numel() < 10:
        return 1.0, 0.0
    A = torch.stack([s_vals, torch.ones_like(s_vals)], dim=1)
    b = t_vals.unsqueeze(1)
    try:
        solution = torch.linalg.lstsq(A, b, driver='gels').solution
        scale = solution[0].item()
        shift = solution[1].item()
    except Exception as e:
        print(f"[Warning] LSTSQ failed: {e}, fallback to identity.")
        scale, shift = 1.0, 0.0
    # A window may not flip or abruptly rescale the depth of the previous one.
    scale = min(max(scale, 0.1), 5.0)
    return scale, shift


class InverseDepthNormalizer:
    """Convert depth to disparity and normalize it to [norm_min, norm_max] per clip.

    The near and far planes are the 2% and 98% quantiles of the valid
    disparities, estimated on at most `max_quantile_samples` random pixels.
    """

    def __init__(self, norm_min=-1.0, norm_max=1.0, quantiles=(0.02, 0.98), clip=True, max_quantile_samples=1000000):
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = norm_max - norm_min
        self.quantiles = quantiles
        self.clip = clip
        self.max_quantile_samples = max_quantile_samples

    def __call__(self, depth, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(depth).bool()
        valid_mask = valid_mask.bool() & (depth > 1e-6)
        disparity = torch.zeros_like(depth)
        disparity[valid_mask] = 1.0 / depth[valid_mask]
        valid_pixels = disparity[valid_mask]
        if valid_pixels.numel() > self.max_quantile_samples:
            indices = torch.randint(0, valid_pixels.numel(), (self.max_quantile_samples,), device=valid_pixels.device)
            valid_pixels = valid_pixels[indices]
        _min = torch.quantile(valid_pixels, self.quantiles[0])
        _max = torch.quantile(valid_pixels, self.quantiles[1])
        if torch.isclose(_max, _min):
            _max = _min + 1e-6
        disparity = (disparity - _min) / (_max - _min) * self.norm_range + self.norm_min
        if self.clip:
            disparity = torch.clip(disparity, self.norm_min, self.norm_max)
        return disparity


def process_video_mask_3d_padded(mask_5d, spatial_downscale_factor=8, temporal_downscale_factor=4, target_temporal_len=20, channel_num=16):
    """Downsample a (B, C, F, H, W) validity mask to the latent grid.

    A latent is valid only if every pixel it covers is valid.
    """
    original_dtype = mask_5d.dtype
    invalid_mask = (~mask_5d.bool()).float()
    padding_needed = max(0, target_temporal_len * temporal_downscale_factor - invalid_mask.shape[2])
    if padding_needed > 0:
        invalid_mask = F.pad(invalid_mask, (0, 0, 0, 0, 0, padding_needed), mode='replicate')
    invalid_mask = F.max_pool3d(
        invalid_mask,
        kernel_size=(temporal_downscale_factor, spatial_downscale_factor, spatial_downscale_factor),
        stride=(temporal_downscale_factor, spatial_downscale_factor, spatial_downscale_factor),
    )
    valid_mask = (~invalid_mask.bool()).to(original_dtype)
    if valid_mask.shape[2] != target_temporal_len:
        print(f"Warning: Output temporal length ({valid_mask.shape[2]}) does not match target ({target_temporal_len}).")
    return valid_mask.repeat(1, channel_num, 1, 1, 1)


def align_dino_features_padded(dino_features, target_temporal_len, temporal_downscale_factor=4, target_hw=None):
    """Average-pool (T, C, h, w) DINOv2 features in time to match the VAE and resize
    them to the token grid. Returns (1, F * H * W, C)."""
    features = dino_features.permute(1, 0, 2, 3).unsqueeze(0)
    padding_needed = max(0, target_temporal_len * temporal_downscale_factor - features.shape[2])
    if padding_needed > 0:
        features = F.pad(features, (0, 0, 0, 0, 0, padding_needed), mode='replicate')
    features = F.avg_pool3d(
        features,
        kernel_size=(temporal_downscale_factor, 1, 1),
        stride=(temporal_downscale_factor, 1, 1),
    )
    if target_hw is not None:
        b, c, t, h, w = features.shape
        features = features.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        features = F.interpolate(features, size=target_hw, mode='bilinear', align_corners=False)
        features = features.view(b, t, c, target_hw[0], target_hw[1]).permute(0, 2, 1, 3, 4)
    if features.shape[2] != target_temporal_len:
        print(f"Warning: DINO temporal length ({features.shape[2]}) does not match target ({target_temporal_len}).")
    return rearrange(features, 'b c f h w -> b (f h w) c')


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        input_params: tuple = None,
        input_params_posi: dict = None,
        input_params_nega: dict = None,
        onload_model_names: tuple = None,
    ):
        self.seperate_cfg = seperate_cfg
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def process(self, pipe, **kwargs) -> dict:
        raise NotImplementedError("`process` is not implemented.")


class PipelineUnitRunner:
    def __call__(self, unit: PipelineUnit, pipe, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict):
        if unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega


class ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device"))

    def process(self, pipe, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        noise = pipe.generate_noise((1, 16, length, height // 8, width // 8), seed=seed, rand_device=rand_device)
        return {"noise": noise}


class DepthVideoEmbedder(PipelineUnit):
    """Encode the normalized ground-truth disparity into the target latents (training)."""

    def __init__(self):
        super().__init__(
            input_params=("depth_video", "mask_video", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, depth_video, mask_video, noise, tiled, tile_size, tile_stride):
        if depth_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        depth_video = pipe.preprocess_depth_video(depth_video.unsqueeze(1), valid_mask=mask_video.unsqueeze(1))
        depth_video = depth_video.to(dtype=pipe.torch_dtype, device=pipe.device)
        depth_video = torch.concat([depth_video, depth_video, depth_video], dim=1)
        input_latents = pipe.vae.encode(depth_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        input_latents = input_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"latents": noise, "input_latents": input_latents}


class MaskVideoEmbedder(PipelineUnit):
    """Downsample the valid-depth mask to the latent grid for the masked loss (training)."""

    def __init__(self):
        super().__init__(input_params=("mask_video", "noise"))

    def process(self, pipe, mask_video, noise):
        if mask_video is None or not pipe.scheduler.training:
            return {"mask_video": None}
        mask_video = repeat(mask_video.unsqueeze(0), "C F H W -> B C F H W", B=1)
        mask_video = process_video_mask_3d_padded(mask_video, target_temporal_len=noise.shape[2])
        return {"mask_video": mask_video}


class DinoFeatureEmbedder(PipelineUnit):
    """Extract DINOv2 features for SRFM and align them to the depth tokens."""

    empty_cache_interval = 10

    def __init__(self):
        super().__init__(input_params=("noise", "video_frames"))
        self.step_count = 0

    def process(self, pipe, noise, video_frames):
        self.step_count += 1
        extractor = pipe.dino_extractor
        if next(extractor.model.parameters()).device != noise.device:
            extractor.model.to(noise.device)
        with torch.autocast("cuda", dtype=pipe.torch_dtype, enabled=noise.device.type == "cuda"):
            dino_feature = extractor.extract(video_frames)
        _, _, f, h, w = noise.shape
        dino_feature = dino_feature.to(dtype=pipe.torch_dtype, device=pipe.device)
        dino_feature = align_dino_features_padded(dino_feature, target_temporal_len=f, target_hw=(h // 2, w // 2))
        if torch.cuda.is_available() and self.step_count % self.empty_cache_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()
        return {"dino_feature": dino_feature}


class PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe, prompt, positive):
        pipe.load_models_to_device(self.onload_model_names)
        return {"context": pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)}


class ReferenceVideoEmbedder(PipelineUnit):
    """Encode the RGB video into the clean in-context condition latents."""

    def __init__(self):
        super().__init__(
            input_params=("reference_video", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, reference_video, num_frames, tiled, tile_size, tile_stride):
        pipe.load_models_to_device(self.onload_model_names)
        reference_video = pipe.preprocess_video(reference_video)
        length = reference_video.shape[2]
        if length < num_frames:
            padding = reference_video[:, :, -1:].repeat(1, 1, num_frames - length, 1, 1)
            reference_video = torch.cat([reference_video, padding], dim=2)
        elif length > num_frames:
            reference_video = reference_video[:, :, :num_frames]
        reference_latents = pipe.vae.encode(reference_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"reference_latents": reference_latents.to(dtype=pipe.torch_dtype, device=pipe.device)}


class ICDepthPipeline(torch.nn.Module):
    """Video depth estimation with a Wan2.1 DiT conditioned in context on the RGB video."""

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, dino_checkpoint=DEFAULT_DINO_CHECKPOINT):
        super().__init__()
        # The device and torch_dtype are used for intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.time_division_factor = 4
        self.time_division_remainder = 1
        self.vram_management_enabled = False
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter()
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.dino_extractor = DINOv2FeatureExtractor(dino_checkpoint)
        self.depth_normalizer = InverseDepthNormalizer()
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            ShapeChecker(),
            NoiseInitializer(),
            DepthVideoEmbedder(),
            MaskVideoEmbedder(),
            DinoFeatureEmbedder(),
            PromptEmbedder(),
            ReferenceVideoEmbedder(),
        ]

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames):
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
            print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
        return height, width, num_frames

    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image

    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def preprocess_depth_video(self, depth, valid_mask):
        """(T, 1, H, W) metric depth -> (1, 1, T, H, W) disparity normalized to [-1, 1]."""
        depth = self.depth_normalizer(depth.to(self.device), valid_mask=valid_mask.to(self.device))
        return rearrange(depth, "T C H W -> 1 C T H W")

    def vae_output_to_disparity(self, vae_output):
        frames = reduce(vae_output, "B C T H W -> T H W C", reduction="mean")
        disparity = (frames.mean(dim=-1) - (-1)) / 2
        return disparity.detach().to(torch.float32).cpu().numpy()

    def vae_output_to_depth_video(self, vae_output):
        frames = reduce(vae_output, "B C T H W -> T H W C", reduction="mean")
        gray = ((frames.mean(dim=-1) - (-1)) / 2 * 255.0).detach().to(torch.float32).cpu().numpy().astype(np.uint8)
        colored = [cv2.cvtColor(cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB) for frame in gray]
        return np.stack(colored, axis=0)

    def load_models_to_device(self, model_names=[]):
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)

    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

    def get_vram(self):
        return torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)

    def freeze_except(self, model_names):
        for name, model in self.named_children():
            if name in model_names:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)

    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, dino_checkpoint=DEFAULT_DINO_CHECKPOINT):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = ICDepthPipeline(device=device, torch_dtype=torch_dtype, dino_checkpoint=dino_checkpoint)
        pipe.fetch_models(model_manager)
        return pipe

    def training_loss(self, **inputs):
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        latents = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        noise_pred = self.dit(
            latents=latents,
            timestep=timestep,
            context=inputs["context"],
            reference_latents=inputs["reference_latents"],
            dino_feature=inputs["dino_feature"],
            use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
            use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
        )
        mask_video = inputs.get("mask_video")
        if mask_video is not None:
            mask = mask_video.to(noise_pred.device, dtype=torch.bool)
            loss = torch.nn.functional.mse_loss(noise_pred[mask].float(), training_target[mask].float())
        else:
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss.float()

    def _predict_noise(self, latents, timestep, inputs_shared, inputs_posi, inputs_nega, kv_cache):
        noise_pred_posi = self.dit(
            latents=latents,
            timestep=timestep,
            context=inputs_posi["context"],
            reference_latents=inputs_shared["reference_latents"],
            dino_feature=inputs_shared["dino_feature"],
            kv_cache=kv_cache,
        )
        cfg_scale = inputs_shared["cfg_scale"]
        if cfg_scale == 1.0:
            return noise_pred_posi
        noise_pred_nega = self.dit(
            latents=latents,
            timestep=timestep,
            context=inputs_nega["context"],
            reference_latents=inputs_shared["reference_latents"],
            dino_feature=inputs_shared["dino_feature"],
            kv_cache=kv_cache,
        )
        return noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

    def _denoise(self, inputs_shared, inputs_posi, inputs_nega, kv_cache=None, progress_bar_cmd=tqdm):
        latents = inputs_shared["latents"]
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self._predict_noise(latents, timestep, inputs_shared, inputs_posi, inputs_nega, kv_cache)
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
        return latents

    def _denoise_sliding_window(self, inputs_shared, inputs_posi, inputs_nega, window_size, overlap, num_inference_steps, sigma_shift, progress_bar_cmd=tqdm):
        """Denoise overlapping temporal windows and blend them in latent space.

        Each window after the first is aligned to the previous one with a
        least-squares scale and shift on the overlap, then linearly blended.
        """
        latents = inputs_shared["latents"]
        reference_latents = inputs_shared["reference_latents"]
        _, _, num_latent_frames, h, w = reference_latents.shape
        dino_feature = rearrange(inputs_shared["dino_feature"], "b (f h w) c -> b f h w c", f=num_latent_frames, h=h // 2, w=w // 2)
        window = (window_size - 1) // 4
        overlap = overlap // 4
        stride = window - overlap
        if stride < 1:
            raise ValueError("`overlap` must be smaller than `window_size`.")

        output = torch.zeros_like(reference_latents)
        for start in range(0, num_latent_frames, stride):
            end = min(start + window, num_latent_frames)
            if start > 0 and end - start <= overlap:
                break
            window_inputs = dict(inputs_shared)
            window_inputs["latents"] = latents[:, :, start:end].clone()
            window_inputs["reference_latents"] = reference_latents[:, :, start:end].clone()
            window_inputs["dino_feature"] = rearrange(dino_feature[:, start:end], "b f h w c -> b (f h w) c")
            self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
            result = self._denoise(window_inputs, inputs_posi, inputs_nega, progress_bar_cmd=progress_bar_cmd)
            if start == 0:
                output[:, :, start:end] = result
            else:
                weights = torch.linspace(0, 1, overlap, device=self.device, dtype=self.torch_dtype).view(1, 1, -1, 1, 1)
                previous = output[:, :, start:start + overlap]
                scale, shift = align_depth_least_square_torch(previous, result[:, :, :overlap])
                result = result * scale + shift
                output[:, :, start:start + overlap] = previous * (1 - weights) + result[:, :, :overlap] * weights
                output[:, :, start + overlap:end] = result[:, :, overlap:]
        return output

    @torch.no_grad()
    def __call__(
        self,
        video,
        height: int,
        width: int,
        num_frames: Optional[int] = None,
        prompt: str = "Estimate the depth",
        negative_prompt: str = "",
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        cfg_scale: float = 5.0,
        num_inference_steps: int = 5,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple = (30, 52),
        tile_stride: tuple = (15, 26),
        use_kv_cache: bool = False,
        window_size: Optional[int] = None,
        overlap: int = 64,
        progress_bar_cmd=tqdm,
    ):
        """Estimate depth for RGB frames that are already resized to (height, width).

        Args:
            video: Sequence of PIL images.
            num_frames: Number of frames processed by the model. Defaults to the
                video length; it is rounded up to 4n+1 by repeating the last frame.
            use_kv_cache: Cache the RGB keys/values after the first step.
            window_size, overlap: Denoise videos longer than `window_size` frames
                in overlapping windows.

        Returns:
            depth_vis: uint8 array (T, H, W, 3), disparity rendered with the inferno colormap.
            disparity: float32 array (T, H, W), relative (affine-invariant) disparity in [0, 1].
        """
        frames = [video[i] for i in range(len(video))]
        if num_frames is None:
            num_frames = len(frames)
        frames = frames[:num_frames]

        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "reference_video": frames,
            "video_frames": np.stack([np.array(frame) for frame in frames]),
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(["dit"])
        if window_size is not None and inputs_shared["num_frames"] > window_size:
            latents = self._denoise_sliding_window(
                inputs_shared, inputs_posi, inputs_nega, window_size, overlap,
                num_inference_steps, sigma_shift, progress_bar_cmd=progress_bar_cmd,
            )
        else:
            kv_cache = [{} for _ in self.dit.blocks] if use_kv_cache else None
            latents = self._denoise(inputs_shared, inputs_posi, inputs_nega, kv_cache, progress_bar_cmd)

        self.load_models_to_device(["vae"])
        vae_output = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        self.load_models_to_device([])
        depth_vis = self.vae_output_to_depth_video(vae_output)[:len(frames)]
        disparity = self.vae_output_to_disparity(vae_output)[:len(frames)]
        return depth_vis, disparity
