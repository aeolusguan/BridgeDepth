"""Stereo clips from full-resolution SceneFlow (not FlyingThings3D subset).

Reference: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
The collection root contains FlyingThings3D/, Monkaa/, Driving/, each with
frames_{final,clean}pass/, disparity/, and optionally camera_data/.
FlyingThings3D has TRAIN/TEST below each modality; Monkaa and Driving have
no official train/test partition and are accepted only with split='train'.
"""

from pathlib import Path

import numpy as np

from .common import StereoClipDataset, StereoFrame
from .io import read_sceneflow_poses


class SceneFlowClips(StereoClipDataset):
    """Dense depth/disparity supervision does not require camera_data.txt.

    Original calibration: 960x540, baseline=1 Blender unit, focal=1050,
    or 450 for Driving/15mm_focallength. The data defines no metre scale.
    Pre-resized releases are deliberately rejected rather than miscalibrated.
    Use StereoClipTransform for calibrated resizing after loading.
    Training uses consecutive forward and reversed clips, following
    DynamicStereo and Stereo Any Video.
    """

    def __init__(self, root, *, split="train", subsets=("FlyingThings3D", "Monkaa", "Driving"),
                 render_pass="final", clip_length=5,
                 load_poses=True, transform=None):
        self.split = split
        root = Path(root).expanduser()
        if split not in ("train", "test"):
            raise ValueError("SceneFlow split must be 'train' or 'test'")
        if render_pass not in ("final", "clean"):
            raise ValueError("render_pass must be 'final' or 'clean'")
        if isinstance(subsets, str):
            subsets = (subsets,)
        if not subsets or len(set(subsets)) != len(subsets):
            raise ValueError("subsets must contain distinct SceneFlow subset names")
        sequences = []
        for subset in subsets:
            if subset not in ("FlyingThings3D", "Monkaa", "Driving"):
                raise ValueError(f"Unknown SceneFlow subset: {subset}")
            if split == "test" and subset != "FlyingThings3D":
                raise ValueError("Only FlyingThings3D has an official test split; "
                                 "set subsets=('FlyingThings3D',) for split='test'")
            subset_root = root if root.name.lower() == subset.lower() else root / subset
            image_root = subset_root / f"frames_{render_pass}pass"
            search_root = image_root / split.upper() if subset == "FlyingThings3D" else image_root
            if not search_root.is_dir():
                raise FileNotFoundError(f"Missing full-release SceneFlow image directory: {search_root}")
            for left_dir in sorted(search_root.rglob("left")):
                if not left_dir.is_dir():
                    continue
                relative = left_dir.parent.relative_to(image_root)
                image_paths = sorted(left_dir.glob("*.png"))
                if not image_paths:
                    continue
                camera_path = subset_root / "camera_data" / relative / "camera_data.txt"
                poses = read_sceneflow_poses(camera_path) if load_poses and camera_path.is_file() else {}
                focal = 450.0 if subset == "Driving" and "15mm_focallength" in relative.parts else 1050.0
                intrinsics = np.array([[focal, 0, 479.5], [0, focal, 269.5], [0, 0, 1]], np.float32)
                stereo_intrinsics = np.stack([intrinsics, intrinsics])
                frames = []
                for left_path in image_paths:
                    try:
                        frame_id = int(left_path.stem)
                    except ValueError as error:
                        raise ValueError(f"Expected numeric SceneFlow frame name: {left_path}") from error
                    right_path = left_dir.parent / "right" / left_path.name
                    disparity_path = subset_root / "disparity" / relative / "left" / f"{left_path.stem}.pfm"
                    for required in (right_path, disparity_path):
                        if not required.is_file():
                            raise FileNotFoundError(f"Incomplete SceneFlow stereo frame: {required}")
                    frames.append(StereoFrame(
                        frame_id, (left_path, right_path), stereo_intrinsics, (540, 960),
                        baseline=1.0, baseline_valid=True, disparity_path=disparity_path,
                        pose=poses.get(frame_id, {}).get("L")))
                sequences.append((f"{subset}/{relative.as_posix()}", frames))
        super().__init__(sequences, dataset="SceneFlow", depth_unit="blender_unit", metric_scale=False,
                         clip_length=clip_length, transform=transform)

    def _clip_indices(self, sequence_length):
        if self.split != "train":
            yield from super()._clip_indices(sequence_length)
            return
        # Match the reference loaders' forward/reverse windows and end boundary.
        for start in range(sequence_length - self.clip_length):
            yield range(start, start + self.clip_length)
            yield range(sequence_length - start - 1,
                        sequence_length - start - self.clip_length - 1, -1)
