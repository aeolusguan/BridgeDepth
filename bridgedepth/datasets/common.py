"""Shared stereo clip assembly and rectification-preserving transforms."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .io import read_float16_depth_png, read_pfm, read_rgb


@dataclass
class StereoFrame:
    frame_id: int
    images: tuple[Path, Path]
    intrinsics: np.ndarray
    size: tuple[int, int]
    baseline: float = 0.0
    baseline_valid: bool = False
    disparity_path: Optional[Path] = None
    depth_path: Optional[Path] = None
    depth_mask_path: Optional[Path] = None
    depth_scale: float = 1.0
    pose: Optional[np.ndarray] = None


class StereoClipDataset(Dataset):
    """Assemble stereo clips using the dataset's temporal sampling policy.

    A batch dimension is added by DataLoader's default_collate. Every item has
    images [T,2,3,H,W] in [0,1]; left disparity/depth and boolean masks
    [T,1,H,W]; intrinsics [T,2,3,3]; baseline/baseline_valid [T]; poses
    [T,4,4] and pose_valid [T]. Poses are OpenCV camera-to-world in the original
    dataset world. Missing poses are identity with pose_valid=False.
    Disparity is x_left-x_right in pixels, depth is camera-axis Z.
    """

    def __init__(self, sequences, *, dataset, depth_unit, metric_scale,
                 clip_length=5, transform=None):
        if not isinstance(clip_length, int) or isinstance(clip_length, bool) or clip_length < 1:
            raise ValueError("clip_length must be a positive integer")
        self.clip_length = clip_length
        self.transform: Optional[Callable] = transform
        self.dataset, self.depth_unit, self.metric_scale = dataset, depth_unit, metric_scale
        self.sequences = []
        self.clips = []
        for scene_id, frames in sequences:
            frames = sorted(frames, key=lambda frame: frame.frame_id)
            # Do not silently bridge a missing frame or a sequence boundary.
            runs = []
            for frame in frames:
                if runs and frame.frame_id == runs[-1][-1].frame_id:
                    raise ValueError(f"Duplicate frame {frame.frame_id} in {scene_id}")
                if not runs or frame.frame_id != runs[-1][-1].frame_id + 1:
                    runs.append([])
                runs[-1].append(frame)
            for run in runs:
                sequence_index = len(self.sequences)
                self.sequences.append((scene_id, run))
                self.clips.extend((sequence_index, indices)
                                  for indices in self._clip_indices(len(run)))
        if not self.clips:
            raise ValueError(f"No {clip_length}-frame clips found for {dataset} "
                             "using its sampling policy; check roots, split and sequence lengths")

    def _clip_indices(self, sequence_length):
        """Chronological, non-overlapping complete clips for evaluation."""
        for start in range(0, sequence_length - self.clip_length + 1, self.clip_length):
            yield range(start, start + self.clip_length)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        sequence_index, indices = self.clips[index]
        scene_id, sequence = self.sequences[sequence_index]
        frames = [sequence[index] for index in indices]
        images, depths, disparities, depth_masks, disparity_masks = [], [], [], [], []
        for frame in frames:
            rgb = [read_rgb(path) for path in frame.images]
            if any(image.shape[:2] != frame.size for image in rgb):
                raise ValueError(f"Image size does not match calibration {frame.size}: {frame.images}")
            images.append(torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2))
            depth = np.zeros(frame.size, dtype=np.float32)
            disparity = np.zeros_like(depth)
            depth_valid = np.zeros(frame.size, dtype=bool)
            disparity_valid = np.zeros_like(depth_valid)
            fx = frame.intrinsics[0, 0, 0]
            offset = frame.intrinsics[0, 0, 2] - frame.intrinsics[1, 0, 2]
            if frame.disparity_path is not None:
                disparity = read_pfm(frame.disparity_path)
                if disparity.shape != frame.size:
                    raise ValueError(f"Disparity size does not match RGB: {frame.disparity_path}")
                disparity_valid = np.isfinite(disparity) & (disparity - offset > 0)
                if frame.baseline_valid:
                    depth_valid = disparity_valid.copy()
                    np.divide(fx * frame.baseline, disparity - offset,
                              out=depth, where=depth_valid)
            if frame.depth_path is not None:
                depth = read_float16_depth_png(frame.depth_path) * frame.depth_scale
                if depth.shape != frame.size:
                    raise ValueError(f"Depth size does not match RGB: {frame.depth_path}")
                depth_valid = np.isfinite(depth) & (depth > 0)
                if frame.depth_mask_path is not None:
                    with Image.open(frame.depth_mask_path) as mask_image:
                        mask = np.asarray(mask_image)
                    if mask.shape != frame.size:
                        raise ValueError(f"Depth mask size does not match RGB: {frame.depth_mask_path}")
                    depth_valid &= mask > 0
                if frame.baseline_valid:
                    disparity_valid = depth_valid.copy()
                    np.divide(fx * frame.baseline, depth, out=disparity, where=depth_valid)
                    disparity[depth_valid] += offset
            depth_valid &= np.isfinite(depth) & (depth > 0)
            disparity_valid &= np.isfinite(disparity)
            depth[~depth_valid] = 0
            disparity[~disparity_valid] = 0
            depths.append(torch.from_numpy(depth)[None])
            disparities.append(torch.from_numpy(disparity)[None])
            depth_masks.append(torch.from_numpy(depth_valid)[None])
            disparity_masks.append(torch.from_numpy(disparity_valid)[None])
        sample = {
            "images": torch.stack(images),
            "disparity": torch.stack(disparities),
            "depth": torch.stack(depths),
            "disparity_valid": torch.stack(disparity_masks),
            "depth_valid": torch.stack(depth_masks),
            "intrinsics": torch.from_numpy(np.stack([f.intrinsics for f in frames])).float(),
            "baseline": torch.tensor([f.baseline for f in frames], dtype=torch.float32),
            "baseline_valid": torch.tensor([f.baseline_valid for f in frames], dtype=torch.bool),
            "poses": torch.from_numpy(np.stack([
                f.pose if f.pose is not None else np.eye(4, dtype=np.float32) for f in frames])).float(),
            "pose_valid": torch.tensor([f.pose is not None for f in frames], dtype=torch.bool),
            "frame_ids": torch.tensor([f.frame_id for f in frames], dtype=torch.int64),
            "scene_id": scene_id,
            "dataset": self.dataset,
            "depth_unit": self.depth_unit,
            "metric_scale": self.metric_scale,
        }
        return self.transform(sample) if self.transform else sample


class StereoClipTransform:
    """Resize then crop identically across time and both rectified views.

    Sizes are (height, width). RGB uses bilinear align_corners=False; labels
    and validity use nearest-exact to preserve discontinuities. Intrinsics
    follow the same pixel-center convention. Disparity scales horizontally;
    depth, baseline and poses retain their physical units. No view-specific
    spatial jitter or horizontal flip is performed.
    """

    def __init__(self, resize=None, crop_size=None, random_crop=True):
        for name, size in (("resize", resize), ("crop_size", crop_size)):
            if size is not None and (len(size) != 2 or any(int(x) != x or x < 1 for x in size)):
                raise ValueError(f"{name} must be a positive (height, width) pair")
        self.resize = tuple(map(int, resize)) if resize is not None else None
        self.crop_size = tuple(map(int, crop_size)) if crop_size is not None else None
        self.random_crop = random_crop

    def __call__(self, sample):
        result = dict(sample)
        intrinsics = sample["intrinsics"].clone()
        images = sample["images"]
        time, views, channels, height, width = images.shape
        if self.resize is not None and self.resize != (height, width):
            new_height, new_width = self.resize
            sx, sy = new_width / width, new_height / height
            result["images"] = F.interpolate(images.flatten(0, 1), size=self.resize,
                                              mode="bilinear", align_corners=False).reshape(
                                                  time, views, channels, new_height, new_width)
            for key in ("disparity", "depth", "disparity_valid", "depth_valid"):
                value = F.interpolate(sample[key].float(), size=self.resize, mode="nearest-exact")
                result[key] = value.bool() if sample[key].dtype == torch.bool else value
            result["disparity"] = result["disparity"] * sx
            intrinsics[..., 0, 0] *= sx
            intrinsics[..., 1, 1] *= sy
            intrinsics[..., 0, 2] = (intrinsics[..., 0, 2] + 0.5) * sx - 0.5
            intrinsics[..., 1, 2] = (intrinsics[..., 1, 2] + 0.5) * sy - 0.5
            height, width = self.resize
        if self.crop_size is not None:
            crop_height, crop_width = self.crop_size
            if crop_height > height or crop_width > width:
                raise ValueError(f"Crop {self.crop_size} exceeds image size {(height, width)}")
            if self.random_crop:
                top = int(torch.randint(height - crop_height + 1, ()).item())
                left = int(torch.randint(width - crop_width + 1, ()).item())
            else:
                top, left = (height - crop_height) // 2, (width - crop_width) // 2
            for key in ("images", "disparity", "depth", "disparity_valid", "depth_valid"):
                result[key] = result[key][..., top:top + crop_height, left:left + crop_width]
            intrinsics[..., 0, 2] -= left
            intrinsics[..., 1, 2] -= top
        result["intrinsics"] = intrinsics
        return result
