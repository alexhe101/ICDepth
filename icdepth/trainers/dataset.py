import os
import random
import warnings

import imageio
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image


class DepthVideoDataset(torch.utils.data.Dataset):
    """RGB-depth video clips for ICDepth training.

    Each metadata row provides `reference_video` (an RGB video) and `depth_video`
    (a `.npz`/`.npy` array of shape (T, H, W) holding depth in meters). Optional
    columns are `height`/`width` (original resolution, used to pick the preset
    without opening the file) and `prompt`.

    Every clip is resized and center-cropped to the largest preset resolution
    that does not upsample it. The clip length follows a fixed token budget of
    384 x 672 x 77, and the clip start is sampled uniformly.
    """

    preset_specs = [
        {"height": 1056, "width": 1920},
        {"height": 768, "width": 1024, "num_frames": 1},
        {"height": 992, "width": 992},
        {"height": 704, "width": 1280},
        {"height": 352, "width": 1152},
        {"height": 640, "width": 640},
        {"height": 480, "width": 640},
        {"height": 384, "width": 672},
        {"height": 368, "width": 640},
    ]
    baseline_height, baseline_width, baseline_frames = 384, 672, 77
    time_division_factor, time_division_remainder = 4, 1
    video_file_extension = ("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm")
    depth_file_extension = ("npy", "npz")

    def __init__(self, metadata_path, base_path="", repeat=1, max_depth=200.0, min_depth=0.001, default_prompt="Estimate the depth"):
        self.base_path = base_path
        self.repeat = repeat
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.default_prompt = default_prompt
        metadata = pd.read_csv(metadata_path)
        self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        self.specs = self._prepare_presets(self.preset_specs)
        self._total_frames_cache = {}

    def _prepare_presets(self, specs):
        baseline_cost = self.baseline_height * self.baseline_width * self.baseline_frames
        out = []
        for spec in specs:
            height, width = int(spec["height"]), int(spec["width"])
            num_frames = int(spec.get("num_frames", 0))
            if num_frames <= 0:
                num_frames = min(max(1, baseline_cost // (height * width)), self.baseline_frames)
                while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                    num_frames -= 1
            out.append({"height": height, "width": width, "num_frames": num_frames})
        out.sort(key=lambda x: x["height"] * x["width"], reverse=True)
        return out

    def _select_spec(self, row, probe_path):
        height = int(row["height"]) if "height" in row else None
        width = int(row["width"]) if "width" in row else None
        if height is None or width is None:
            try:
                shape = self._load_depth_array(probe_path).shape
                height, width = int(shape[1]), int(shape[2])
            except Exception:
                return self.specs[0]
        candidates = [spec for spec in self.specs if spec["height"] <= height and spec["width"] <= width]
        if not candidates:
            return {
                "height": min(height, self.specs[-1]["height"]),
                "width": min(width, self.specs[-1]["width"]),
                "num_frames": self.specs[-1]["num_frames"],
            }
        return candidates[0]

    def _path(self, path):
        return os.path.join(self.base_path, path)

    def _load_depth_array(self, path):
        if path.endswith(".npz"):
            with np.load(path) as archive:
                keys = list(archive.keys())
                if len(keys) == 0:
                    raise KeyError(f"No arrays found in npz: {path}")
                if len(keys) == 1:
                    return archive[keys[0]]
                for key in ("data", "arr_0", "depth", "mask", "array", "values"):
                    if key in keys:
                        return archive[key]
                return archive[keys[0]]
        return np.load(path, mmap_mode="r")

    def _count_total_frames(self, path):
        if path not in self._total_frames_cache:
            try:
                self._total_frames_cache[path] = int(self._load_depth_array(path).shape[0])
            except Exception:
                self._total_frames_cache[path] = 1
        return self._total_frames_cache[path]

    def _clip_length(self, num_frames, total):
        num_frames = min(int(num_frames), total)
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return torchvision.transforms.functional.center_crop(image, (target_height, target_width))

    def crop_and_resize_depth(self, depth, target_height, target_width):
        """Nearest-neighbor resize and center crop of a (T, H, W) depth tensor."""
        depth = depth.unsqueeze(1)
        height, width = depth.shape[2], depth.shape[3]
        scale = max(target_width / width, target_height / height)
        depth = torchvision.transforms.functional.resize(
            depth,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            antialias=False,
        )
        depth = torchvision.transforms.functional.center_crop(depth, (target_height, target_width))
        return depth.squeeze(1)

    def load_video(self, path, height, width, num_frames, start_index):
        try:
            reader = imageio.get_reader(path)
        except (OSError, IOError) as e:
            warnings.warn(f"load_video failed for {path!r}: {e}")
            return None
        try:
            total = self._total_frames_cache.get(path)
            if total is None:
                total = int(reader.count_frames())
                self._total_frames_cache[path] = total
            num_frames = self._clip_length(num_frames, total)
            start = max(0, min(int(start_index), max(0, total - num_frames)))
            frames = []
            for frame_id in range(start, start + num_frames):
                frame = Image.fromarray(reader.get_data(frame_id))
                frames.append(self.crop_and_resize(frame, height, width))
        except (OSError, IOError) as e:
            warnings.warn(f"load_video failed for {path!r}: {e}")
            return None
        finally:
            reader.close()
        return frames

    def load_depth(self, path, height, width, num_frames, start_index):
        data = self._load_depth_array(path)
        length = data.shape[0]
        num_frames = self._clip_length(num_frames, length)
        start = max(0, min(int(start_index), max(0, length - num_frames)))
        depth = torch.from_numpy(data[start:start + num_frames].astype(np.float32))
        return self.crop_and_resize_depth(depth, height, width)

    def __getitem__(self, data_id, _skip_retries=0):
        max_skip_retries = 32
        row = self.data[data_id % len(self.data)]
        depth_path = self._path(row["depth_video"])
        video_path = self._path(row["reference_video"])
        spec = self._select_spec(row, depth_path)
        height, width = spec["height"], spec["width"]
        total = self._count_total_frames(depth_path)
        num_frames = self._clip_length(spec["num_frames"], total)
        start_index = random.randint(0, max(0, total - num_frames))

        depth = self.load_depth(depth_path, height, width, num_frames, start_index)
        frames = self.load_video(video_path, height, width, num_frames, start_index)
        if frames is None:
            if _skip_retries >= max_skip_retries:
                raise RuntimeError(f"Too many consecutive load failures (e.g. bad video at {video_path!r}).")
            return self.__getitem__((data_id + 1) % len(self.data), _skip_retries + 1)

        mask = torch.isfinite(depth) & (depth < self.max_depth) & (depth > self.min_depth)
        prompt = row.get("prompt")
        return {
            "prompt": prompt if isinstance(prompt, str) else self.default_prompt,
            "reference_video": frames,
            "video_frames": np.array(frames),
            "depth_video": depth,
            "mask_video": mask,
            "_height": height,
            "_width": width,
            "_num_frames": num_frames,
        }

    def __len__(self):
        return len(self.data) * self.repeat
