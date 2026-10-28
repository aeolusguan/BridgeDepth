"""Dynamic Replica clips from its original gzip JSON annotations and PNGs.

Format references (implementation here is independent):
https://github.com/facebookresearch/dynamic_stereo/blob/main/datasets/dynamic_stereo_datasets.py
https://github.com/TomTomTommi/stereoanyvideo/blob/main/datasets/video_datasets.py
"""

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import StereoClipDataset, StereoFrame


def camera_from_annotation(viewpoint, image_size):
    """Return pixel K and optional OpenCV c2w; no PyTorch3D installation needed.

    PyTorch3D annotations use row-vector world-to-camera X_cam=X_world@R+T,
    with camera axes left/up/forward. Image size is (height, width).
    Intrinsics remain usable when extrinsics R/T are missing.
    """
    if not viewpoint:
        raise ValueError("Dynamic Replica viewpoint intrinsics are required")
    half_wh = np.asarray(image_size[::-1], dtype=np.float64) / 2.0
    convention = viewpoint.get("intrinsics_format", "ndc_norm_image_bounds").lower()
    if convention == "ndc_norm_image_bounds":
        scale = half_wh
    elif convention == "ndc_isotropic":
        scale = half_wh.min()
    else:
        raise ValueError(f"Unknown Dynamic Replica intrinsics_format: {convention}")
    focal = np.asarray(viewpoint["focal_length"], dtype=np.float64) * scale
    principal = half_wh - np.asarray(viewpoint["principal_point"], dtype=np.float64) * scale
    if focal.shape != (2,) or principal.shape != (2,) or not np.isfinite([focal, principal]).all() or (focal <= 0).any():
        raise ValueError("Invalid Dynamic Replica camera intrinsics")
    intrinsic = np.array([[focal[0], 0, principal[0]], [0, focal[1], principal[1]], [0, 0, 1]], np.float32)
    pose = None
    if viewpoint.get("R") is not None and viewpoint.get("T") is not None:
        rotation = np.asarray(viewpoint["R"], dtype=np.float64)
        translation = np.asarray(viewpoint["T"], dtype=np.float64)
        if (rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError("Invalid Dynamic Replica camera extrinsics")
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation @ np.diag([-1.0, -1.0, 1.0])
        pose[:3, 3] = -rotation @ translation
    return intrinsic, pose


def rectified_baseline(intrinsics, left_pose, right_pose):
    """Validate the rectified model and return signed positive baseline."""
    left_k, right_k = intrinsics
    if not np.allclose(left_k[[0, 1, 1], [0, 1, 2]], right_k[[0, 1, 1], [0, 1, 2]], atol=1e-3, rtol=1e-5):
        raise ValueError("Stereo pair is not rectified: fx, fy, or cy differs")
    if left_pose is None or right_pose is None:
        return None
    right_to_left = np.linalg.inv(left_pose.astype(np.float64)) @ right_pose
    baseline = float(right_to_left[0, 3])
    tolerance = max(1e-5, abs(baseline) * 1e-3)
    if (not np.allclose(right_to_left[:3, :3], np.eye(3), atol=1e-4)
            or not np.allclose(right_to_left[1:3, 3], 0, atol=tolerance) or baseline <= 0):
        raise ValueError("Stereo extrinsics are not a rectified left/right pair with positive baseline")
    return baseline


class DynamicReplicaClips(StereoClipDataset):
    """Root contains {train,valid,test}/frame_annotations_{split}.jgz.

    A split directory itself is also accepted. Missing extrinsics do not
    disable depth supervision; disparity then requires an explicitly supplied
    baseline in metres. This fallback never manufactures pose labels.
    RGB/depth sizes must match the annotation calibration. Foreground masks
    are intentionally unused: background depth is valid training supervision.
    Training samples a stride from 1–5 at each candidate start, spaced three
    frames apart. The resulting clip list is fixed at dataset construction.
    """

    def __init__(self, root, *, split="train", clip_length=5,
                 baseline=None, load_poses=True, transform=None):
        self.split = split
        if split not in ("train", "valid", "test"):
            raise ValueError("Dynamic Replica split must be 'train', 'valid', or 'test'")
        if baseline is not None and (not np.isfinite(baseline) or baseline <= 0):
            raise ValueError("baseline must be a positive value in metres")
        root = Path(root).expanduser()
        annotation_name = f"frame_annotations_{split}.jgz"
        split_root = root if (root / annotation_name).is_file() else root / split
        annotation_path = split_root / annotation_name
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing Dynamic Replica annotations: {annotation_path}")
        with gzip.open(annotation_path, "rt", encoding="utf8") as stream:
            annotations = json.load(stream)
        if not isinstance(annotations, list):
            raise ValueError(f"Expected a JSON annotation list: {annotation_path}")
        grouped = defaultdict(lambda: defaultdict(dict))
        for entry in annotations:
            camera = entry["camera_name"]
            if camera not in ("left", "right"):
                raise ValueError(f"Unknown Dynamic Replica camera_name: {camera}")
            scene, frame_id = entry["sequence_name"], int(entry["frame_number"])
            if camera in grouped[scene][frame_id]:
                raise ValueError(f"Duplicate Dynamic Replica annotation: {scene}/{frame_id}/{camera}")
            grouped[scene][frame_id][camera] = entry
        sequences = []
        for scene in sorted(grouped):
            frames = []
            for frame_id, pair in sorted(grouped[scene].items()):
                if set(pair) != {"left", "right"}:
                    raise ValueError(f"Missing synchronized stereo view: {scene}/{frame_id}")
                left, right = pair["left"], pair["right"]
                if (left.get("frame_timestamp") is not None and right.get("frame_timestamp") is not None
                        and not np.isclose(left["frame_timestamp"], right["frame_timestamp"], atol=1e-6, rtol=0)):
                    raise ValueError(f"Stereo timestamps disagree: {scene}/{frame_id}")
                size = tuple(left["image"]["size"])
                if len(size) != 2 or min(size) < 1 or size != tuple(right["image"]["size"]):
                    raise ValueError(f"Stereo image dimensions disagree: {scene}/{frame_id}")
                left_k, left_pose = camera_from_annotation(left.get("viewpoint"), size)
                right_k, right_pose = camera_from_annotation(right.get("viewpoint"), size)
                intrinsics = np.stack([left_k, right_k])
                calibrated_baseline = rectified_baseline(intrinsics, left_pose, right_pose)
                if calibrated_baseline is not None and baseline is not None and not np.isclose(
                        calibrated_baseline, baseline, rtol=1e-3, atol=1e-5):
                    raise ValueError(f"Supplied baseline disagrees with calibrated extrinsics: {scene}/{frame_id}")
                frame_baseline = calibrated_baseline if calibrated_baseline is not None else baseline
                depth_annotation = left.get("depth")
                if not depth_annotation or not depth_annotation.get("path"):
                    raise ValueError(f"Missing left depth annotation: {scene}/{frame_id}")
                depth_scale = depth_annotation.get("scale_adjustment", 1.0)
                if depth_scale is None:
                    depth_scale = 1.0
                if not np.isfinite(depth_scale) or depth_scale <= 0:
                    raise ValueError(f"Invalid depth scale_adjustment: {scene}/{frame_id}")
                image_paths = (split_root / left["image"]["path"], split_root / right["image"]["path"])
                depth_path = split_root / depth_annotation["path"]
                mask_path = depth_annotation.get("mask_path")
                depth_mask_path = split_root / mask_path if mask_path else None
                required_paths = [*image_paths, depth_path]
                if depth_mask_path is not None:
                    required_paths.append(depth_mask_path)
                for required in required_paths:
                    if not required.is_file():
                        raise FileNotFoundError(f"Incomplete Dynamic Replica frame: {required}")
                frames.append(StereoFrame(
                    frame_id, image_paths, intrinsics, size,
                    baseline=float(frame_baseline or 0), baseline_valid=frame_baseline is not None,
                    depth_path=depth_path, depth_mask_path=depth_mask_path, depth_scale=depth_scale,
                    pose=left_pose if load_poses else None))
            sequences.append((f"{split}/{scene}", frames))
        super().__init__(sequences, dataset="DynamicReplica", depth_unit="m", metric_scale=True,
                         clip_length=clip_length, transform=transform)

    def _clip_indices(self, sequence_length):
        if self.split != "train":
            yield from super()._clip_indices(sequence_length)
            return
        for start in range(0, sequence_length, 3):
            stride = 1 if self.clip_length == 1 else np.random.randint(1, 6)
            if start + stride * self.clip_length < sequence_length:
                yield range(start, start + stride * self.clip_length, stride)
