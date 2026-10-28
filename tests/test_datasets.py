"""Format and geometry fixtures; no external dataset downloads are needed."""

import gzip
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import default_collate

from bridgedepth.datasets import DynamicReplicaClips, SceneFlowClips, StereoClipTransform
from bridgedepth.datasets.dynamic_replica import camera_from_annotation, rectified_baseline
from bridgedepth.datasets.io import read_float16_depth_png, read_pfm, read_sceneflow_poses


def write_pfm(path, values, little_endian=True, scale=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = values.shape
    with path.open("wb") as stream:
        stream.write(f"Pf\n{width} {height}\n{(-scale if little_endian else scale)}\n".encode())
        stream.write(np.flipud(values / scale).astype("<f4" if little_endian else ">f4").tobytes())


def write_image(path, size=(4, 6), value=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((*size, 3), value, dtype=np.uint8)).save(path)


def write_depth(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, np.float16).view(np.uint16)).save(path)


def make_dynamic_replica(root, scenes=None, missing_poses=False, scale=1.0, split="train"):
    if scenes is None:
        scenes = {"sequence": list(range(12))}
    split_root = root / split
    annotations = []
    for scene, frame_ids in scenes.items():
        for frame_id in frame_ids:
            for camera in ("left", "right"):
                image_path = f"{scene}/{camera}/{frame_id}.png"
                depth_path = f"{scene}/{camera}/{frame_id}_depth.png"
                write_image(split_root / image_path, value=frame_id + 32)
                write_depth(split_root / depth_path, np.full((4, 6), 2.0))
                viewpoint = {"focal_length": [2.0, 2.0], "principal_point": [0.0, 0.0],
                             "intrinsics_format": "ndc_isotropic"}
                if not missing_poses:
                    viewpoint.update(R=np.eye(3).tolist(), T=[0.2 if camera == "right" else 0.0, 0.0, frame_id * 0.01])
                annotations.append({"sequence_name": scene, "frame_number": frame_id,
                                    "frame_timestamp": frame_id / 30, "camera_name": camera,
                                    "image": {"path": image_path, "size": [4, 6]},
                                    "depth": {"path": depth_path, "scale_adjustment": scale},
                                    "viewpoint": viewpoint})
    # Deliberately scramble temporal and view ordering in the annotation file.
    with gzip.open(split_root / f"frame_annotations_{split}.jgz", "wt") as stream:
        json.dump(list(reversed(annotations)), stream)


def make_sceneflow(root, subset="FlyingThings3D", sequence="TRAIN/A/0000", frame_ids=(6, 7, 8), with_pose=False):
    subset_root = root / subset
    for frame_id in frame_ids:
        for camera in ("left", "right"):
            write_image(subset_root / "frames_finalpass" / sequence / camera / f"{frame_id:04}.png", size=(540, 960))
        write_pfm(subset_root / "disparity" / sequence / "left" / f"{frame_id:04}.pfm",
                  np.full((540, 960), 10.0, dtype=np.float32))
    if with_pose:
        camera_path = subset_root / "camera_data" / sequence / "camera_data.txt"
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        with camera_path.open("w") as stream:
            for frame_id in frame_ids:
                stream.write(f"Frame {frame_id:04}\n")
                for camera in ("L", "R"):
                    pose = np.eye(4)
                    pose[0, 3] = float(camera == "R")
                    pose[2, 3] = frame_id
                    stream.write(camera + " " + " ".join(map(str, pose.ravel())) + "\n")
    return subset_root


@pytest.mark.parametrize("little_endian", [True, False])
def test_pfm_endianness_vertical_flip_scale(tmp_path, little_endian):
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
    path = tmp_path / "depth.pfm"
    write_pfm(path, expected, little_endian=little_endian, scale=2.0)
    np.testing.assert_array_equal(read_pfm(path), expected)


def test_float16_png_bits(tmp_path):
    expected = np.array([[1.5, 2.25, 0], [10, np.inf, np.nan]], np.float32)
    path = tmp_path / "depth.png"
    write_depth(path, expected)
    np.testing.assert_allclose(read_float16_depth_png(path), expected, equal_nan=True)


def test_dynamic_replica_reference_training_sampling_is_fixed(tmp_path):
    make_dynamic_replica(tmp_path, {"sequence": list(range(30))})
    np.random.seed(7)
    data = DynamicReplicaClips(tmp_path)
    # Seed 7 draws strides 5,2,4,4,5,2,1,2,3,3 at starts 0,3,...,27.
    # The reference accepts a candidate only when start + stride*T < length.
    expected = [[0, 5, 10, 15, 20], [3, 5, 7, 9, 11], [6, 10, 14, 18, 22],
                [9, 13, 17, 21, 25], [15, 17, 19, 21, 23], [18, 19, 20, 21, 22]]
    assert [data[i]["frame_ids"].tolist() for i in range(len(data))] == expected
    clips = list(data.clips)
    np.random.seed(99)
    assert [data[i]["frame_ids"].tolist() for i in range(len(data))] == expected
    assert data.clips == clips


@pytest.mark.parametrize("split,clip_length,expected", [
    ("train", 1, [[0], [3], [6], [0], [5], [8]]),
    ("test", 2, [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1], [5, 6], [7, 8], [9, 10]]),
])
def test_dynamic_replica_clips_do_not_cross_gaps_or_scenes(tmp_path, split, clip_length, expected):
    make_dynamic_replica(tmp_path, {"a": list(range(8)), "b": [0, 1, 2, 5, 6, 7, 8, 9, 10]}, split=split)
    data = DynamicReplicaClips(tmp_path, split=split, clip_length=clip_length)
    samples = [data[i] for i in range(len(data))]
    assert [sample["frame_ids"].tolist() for sample in samples] == expected
    first_scene_count = 3 if split == "train" else 4
    assert samples[first_scene_count - 1]["scene_id"] != samples[first_scene_count]["scene_id"]
    if split == "test":
        # Evaluation does not consume random strides or reverse time.
        np.random.seed(123)
        again = DynamicReplicaClips(tmp_path, split=split, clip_length=clip_length)
        assert again.clips == data.clips


def test_dynamic_replica_contract_missing_poses_and_collation(tmp_path):
    present = tmp_path / "present"
    absent = tmp_path / "absent"
    make_dynamic_replica(present)
    make_dynamic_replica(absent, missing_poses=True)
    a = DynamicReplicaClips(present, clip_length=2)[0]
    b = DynamicReplicaClips(absent, clip_length=2)[0]
    assert a["images"].shape == (2, 2, 3, 4, 6)
    assert a["images"].dtype == torch.float32
    assert 0 <= a["images"].min() <= a["images"].max() <= 1
    assert a["disparity_valid"].dtype == torch.bool
    assert a["pose_valid"].all() and a["baseline_valid"].all()
    torch.testing.assert_close(a["baseline"], torch.full((2,), 0.2))
    torch.testing.assert_close(a["disparity"], torch.full((2, 1, 4, 6), 0.4))
    assert b["depth_valid"].all()
    assert not b["pose_valid"].any() and not b["baseline_valid"].any()
    assert not b["disparity_valid"].any()
    torch.testing.assert_close(b["poses"], torch.eye(4).repeat(2, 1, 1))
    batch = default_collate([a, b])
    assert batch["images"].shape == (2, 2, 2, 3, 4, 6)
    assert batch["pose_valid"].tolist() == [[True, True], [False, False]]
    fallback = DynamicReplicaClips(absent, clip_length=2, baseline=0.2)[0]
    assert fallback["disparity_valid"].all() and not fallback["pose_valid"].any()


def test_dynamic_replica_depth_scale_and_nonfinite_labels(tmp_path):
    make_dynamic_replica(tmp_path, scale=2)
    depth_path = tmp_path / "train/sequence/left/0_depth.png"
    write_depth(depth_path, np.array([[0, np.inf, np.nan, 2, 2, 2]] * 4))
    sample = DynamicReplicaClips(tmp_path, clip_length=1)[0]
    assert not sample["depth_valid"][0, 0, :, :3].any()
    assert torch.isfinite(sample["depth"]).all() and torch.isfinite(sample["disparity"]).all()
    torch.testing.assert_close(sample["depth"][0, 0, :, 3:], torch.full((4, 3), 4.0))


def test_dynamic_replica_explicit_depth_mask(tmp_path):
    make_dynamic_replica(tmp_path)
    annotation_path = tmp_path / "train/frame_annotations_train.jgz"
    with gzip.open(annotation_path, "rt") as stream:
        annotations = json.load(stream)
    mask = np.full((4, 6), 255, dtype=np.uint8)
    mask[:, 0] = 0
    Image.fromarray(mask).save(tmp_path / "train/mask.png")
    for entry in annotations:
        entry["depth"]["mask_path"] = "mask.png"
    with gzip.open(annotation_path, "wt") as stream:
        json.dump(annotations, stream)
    sample = DynamicReplicaClips(tmp_path, clip_length=1)[0]
    assert not sample["depth_valid"][..., 0].any()
    assert not sample["disparity_valid"][..., 0].any()
    assert sample["depth_valid"][..., 1:].all()


def test_dynamic_replica_rejects_missing_stereo_frame(tmp_path):
    make_dynamic_replica(tmp_path)
    annotation_path = tmp_path / "train/frame_annotations_train.jgz"
    with gzip.open(annotation_path, "rt") as stream:
        annotations = json.load(stream)
    annotations = [entry for entry in annotations
                   if not (entry["frame_number"] == 1 and entry["camera_name"] == "right")]
    with gzip.open(annotation_path, "wt") as stream:
        json.dump(annotations, stream)
    with pytest.raises(ValueError, match="synchronized stereo"):
        DynamicReplicaClips(tmp_path, clip_length=2)


@pytest.mark.parametrize("convention,expected_focal", [("ndc_isotropic", [4, 6]), ("ndc_norm_image_bounds", [6, 6])])
def test_pytorch3d_conversion_and_projection(convention, expected_focal):
    # A rotated rig catches transpose mistakes that an identity rig cannot.
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    left = {"R": rotation.tolist(), "T": [0.3, -0.1, 0.5], "focal_length": [2, 3],
            "principal_point": [0.1, -0.2], "intrinsics_format": convention}
    right = dict(left, T=[0.5, -0.1, 0.5], principal_point=[0.2, -0.2])
    kl, cl = camera_from_annotation(left, (4, 6))
    kr, cr = camera_from_annotation(right, (4, 6))
    np.testing.assert_allclose([kl[0, 0], kl[1, 1]], expected_focal)
    baseline = rectified_baseline(np.stack([kl, kr]), cl, cr)
    assert baseline == pytest.approx(0.2)
    pixel = np.array([3.0, 2.0, 1.0])
    xyz_left = np.linalg.inv(kl) @ pixel * 2.5
    xyz_world = cl @ np.r_[xyz_left, 1]
    xyz_right = np.linalg.inv(cr) @ xyz_world
    projected = kr @ xyz_right[:3]
    projected /= projected[2]
    expected_disparity = kl[0, 0] * baseline / 2.5 + kl[0, 2] - kr[0, 2]
    assert projected[1] == pytest.approx(pixel[1], abs=1e-5)
    assert pixel[0] - projected[0] == pytest.approx(expected_disparity, abs=1e-5)


def test_rectification_rejects_vertical_rig_offset():
    left = np.eye(4, dtype=np.float32)
    right = left.copy()
    right[:3, 3] = [0.2, 0.01, 0]
    intrinsics = np.stack([np.eye(3), np.eye(3)])
    with pytest.raises(ValueError, match="rectified"):
        rectified_baseline(intrinsics, left, right)


def test_shared_resize_crop_preserves_geometry(tmp_path):
    make_dynamic_replica(tmp_path)
    sample = DynamicReplicaClips(tmp_path, clip_length=2)[0]
    original_k = sample["intrinsics"].clone()
    transformed = StereoClipTransform(resize=(8, 18), crop_size=(6, 14), random_crop=False)(sample)
    assert transformed["images"].shape == (2, 2, 3, 6, 14)
    torch.testing.assert_close(transformed["intrinsics"][..., 0, 0], original_k[..., 0, 0] * 3)
    torch.testing.assert_close(transformed["intrinsics"][..., 1, 1], original_k[..., 1, 1] * 2)
    torch.testing.assert_close(transformed["intrinsics"][..., 0, 2], (original_k[..., 0, 2] + 0.5) * 3 - 0.5 - 2)
    torch.testing.assert_close(transformed["intrinsics"][..., 1, 2], (original_k[..., 1, 2] + 0.5) * 2 - 0.5 - 1)
    torch.testing.assert_close(transformed["depth"], torch.full((2, 1, 6, 14), 2.0))
    torch.testing.assert_close(transformed["disparity"], torch.full((2, 1, 6, 14), 1.2))
    torch.testing.assert_close(transformed["poses"], sample["poses"])
    torch.testing.assert_close(sample["intrinsics"], original_k)  # no mutation
    torch.testing.assert_close(transformed["images"][:, 0], transformed["images"][:, 1])


@pytest.mark.parametrize("subset,sequence,focal", [
    ("FlyingThings3D", "TRAIN/A/0000", 1050),
    ("Monkaa", "example_scene", 1050),
    ("Driving", "15mm_focallength/scene_forwards/fast", 450),
])
def test_sceneflow_full_release_depth_without_pose(tmp_path, subset, sequence, focal):
    subset_root = make_sceneflow(tmp_path, subset, sequence)
    sample = SceneFlowClips(subset_root, subsets=(subset,), clip_length=2)[0]
    assert sample["frame_ids"].tolist() == [6, 7]
    assert sample["depth_unit"] == "blender_unit" and sample["metric_scale"] is False
    assert not sample["pose_valid"].any() and sample["depth_valid"].all()
    torch.testing.assert_close(sample["depth"], torch.full((2, 1, 540, 960), focal / 10))
    assert sample["intrinsics"][0, 0, 0, 0] == focal


def test_sceneflow_pose_conversion_and_split_isolation(tmp_path):
    make_sceneflow(tmp_path, with_pose=True)
    make_sceneflow(tmp_path, sequence="TEST/A/0000")
    train = SceneFlowClips(tmp_path, subsets=("FlyingThings3D",), clip_length=2)
    test = SceneFlowClips(tmp_path, subsets=("FlyingThings3D",), split="test", clip_length=2)
    assert len(train) == 2 and len(test) == 1
    assert "/TRAIN/" in train[0]["scene_id"] and "/TEST/" in test[0]["scene_id"]
    sample = train[0]
    assert sample["pose_valid"].all()
    torch.testing.assert_close(sample["poses"][0, :3, :3], torch.diag(torch.tensor([1.0, -1.0, -1.0])))
    assert sample["poses"][0, 2, 3] == 6
    records = read_sceneflow_poses(tmp_path / "FlyingThings3D/camera_data/TRAIN/A/0000/camera_data.txt")
    relative = np.linalg.inv(records[6]["L"]) @ records[6]["R"]
    np.testing.assert_allclose(relative[:3, 3], [1, 0, 0])


def test_sceneflow_reference_forward_reverse_windows(tmp_path):
    make_sceneflow(tmp_path, frame_ids=range(6, 16))
    data = SceneFlowClips(tmp_path, subsets=("FlyingThings3D",))
    # Ten-frame sequences yield five five-frame windows in each direction.
    expected = [[6, 7, 8, 9, 10], [15, 14, 13, 12, 11],
                [7, 8, 9, 10, 11], [14, 13, 12, 11, 10],
                [8, 9, 10, 11, 12], [13, 12, 11, 10, 9],
                [9, 10, 11, 12, 13], [12, 11, 10, 9, 8],
                [10, 11, 12, 13, 14], [11, 10, 9, 8, 7]]
    actual = [[data.sequences[sequence][1][index].frame_id for index in indices]
              for sequence, indices in data.clips]
    assert actual == expected


def test_sceneflow_reverses_images_labels_poses_and_ids_together(tmp_path):
    subset_root = make_sceneflow(tmp_path, frame_ids=(6, 7, 8, 9), with_pose=True)
    for frame_id in (6, 7, 8, 9):
        for camera in ("left", "right"):
            write_image(subset_root / "frames_finalpass/TRAIN/A/0000" / camera / f"{frame_id:04}.png",
                        size=(540, 960), value=frame_id)
        write_pfm(subset_root / "disparity/TRAIN/A/0000/left" / f"{frame_id:04}.pfm",
                  np.full((540, 960), frame_id, dtype=np.float32))
    data = SceneFlowClips(tmp_path, subsets=("FlyingThings3D",), clip_length=3)
    reverse = data[1]
    expected_ids = torch.tensor([9, 8, 7])
    torch.testing.assert_close(reverse["frame_ids"], expected_ids)
    torch.testing.assert_close(reverse["images"][:, :, 0, 0, 0], expected_ids[:, None].expand(-1, 2) / 255)
    torch.testing.assert_close(reverse["disparity"][:, 0, 0, 0], expected_ids.float())
    torch.testing.assert_close(reverse["depth"][:, 0, 0, 0], 1050 / expected_ids)
    torch.testing.assert_close(reverse["poses"][:, 2, 3], expected_ids.float())
    assert reverse["pose_valid"].all() and reverse["disparity_valid"].all()


def test_sceneflow_evaluation_is_chronological_nonoverlapping(tmp_path):
    make_sceneflow(tmp_path, sequence="TEST/A/0000", frame_ids=range(6, 13))
    data = SceneFlowClips(tmp_path, subsets=("FlyingThings3D",), split="test", clip_length=3)
    assert [data[i]["frame_ids"].tolist() for i in range(len(data))] == [[6, 7, 8], [9, 10, 11]]


def test_bad_roots_and_split_have_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="full-release"):
        SceneFlowClips(tmp_path)
    with pytest.raises(FileNotFoundError, match="annotations"):
        DynamicReplicaClips(tmp_path)
    with pytest.raises(ValueError, match="official test split"):
        SceneFlowClips(tmp_path, split="test", subsets=("Monkaa",))
