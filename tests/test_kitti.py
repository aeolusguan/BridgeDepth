"""KITTI directory discovery, year selection and image-loader integration."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("cv2")
pytest.importorskip("imageio")
pytest.importorskip("tabulate")

from bridgedepth.config import get_cfg
from bridgedepth.dataloader.datasets import KITTI, build_train_loader, build_val_loader


def make_pairs(root, year, split, ids, labels=False):
    left, right = ("colored_0", "colored_1") if year == "2012" else ("image_2", "image_3")
    names = [left, right]
    if labels:
        names += ["disp_occ", "disp_noc"] if year == "2012" else ["disp_occ_0", "disp_noc_0"]
    directory = root / f"KITTI_{year}" / split
    for name in names:
        (directory / name).mkdir(parents=True, exist_ok=True)
        for index in reversed(list(ids)):
            (directory / name / f"{index:06d}_10.png").touch()
        (directory / name / "000000_11.png").touch()


@pytest.fixture(scope="module")
def kitti_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("kitti") / "datasets" / "KITTI"
    make_pairs(root, "2012", "training", range(194), labels=True)
    make_pairs(root, "2015", "training", range(200), labels=True)
    return root


@pytest.mark.parametrize("year,count", [("2012", 194), ("2015", 200)])
@pytest.mark.parametrize("nonocc", [False, True])
def test_year_discovery_uses_all_labeled_frames(kitti_root, year, count, nonocc):
    dataset = KITTI(root=str(kitti_root), image_set=year, nonocc=nonocc)
    assert len(dataset) == len(dataset.disparity_list) == count
    assert not dataset.is_test
    expected = [f"{index:06d}_10.png" for index in range(count)]
    assert [Path(pair[0]).name for pair in dataset.image_list] == expected
    assert [Path(pair[1]).name for pair in dataset.image_list] == expected
    assert [Path(path).name for path in dataset.disparity_list] == expected
    disparity_dir = "disp_noc" if nonocc else "disp_occ"
    if year == "2015":
        disparity_dir += "_0"
    assert all(Path(path).parent.name == disparity_dir for path in dataset.disparity_list)


@pytest.mark.parametrize("year", ["2012", "2015"])
def test_official_test_images_need_no_disparity(tmp_path, year):
    make_pairs(tmp_path, year, "testing", [9, 2, 0])
    dataset = KITTI(root=str(tmp_path), image_set=year, split="testing")
    assert dataset.is_test
    assert dataset.disparity_list == []
    assert [Path(pair[0]).name for pair in dataset.image_list] == ["000000_10.png", "000002_10.png", "000009_10.png"]


def test_training_and_evaluation_builders_use_full_years(kitti_root, monkeypatch):
    monkeypatch.chdir(kitti_root.parent.parent)
    cfg = get_cfg()
    cfg.DATASETS.TRAIN = ["kitti_mix"]
    cfg.DATASETS.CROP_SIZE = [8, 8]
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 2
    loader, sampler = build_train_loader(cfg)
    assert sampler is None
    assert len(loader.dataset) == 394
    assert [len(dataset) for dataset in loader.dataset.datasets] == [200, 194]
    assert "KITTI_2015" in loader.dataset.datasets[0].image_list[0][0]
    assert "KITTI_2012" in loader.dataset.datasets[1].image_list[0][0]
    for name, count in (("kitti_2012_trainval", 194), ("kitti_2015_trainval", 200)):
        cfg.DATASETS.TRAIN = [name]
        train, _ = build_train_loader(cfg)
        evaluation = build_val_loader(cfg, name)
        assert len(train.dataset) == len(evaluation.dataset) == count


def test_real_png_sample_preserves_image_and_disparity_units(tmp_path):
    make_pairs(tmp_path, "2015", "training", [5], labels=True)
    directory = tmp_path / "KITTI_2015" / "training"
    for name, value in (("image_2", 10), ("image_3", 30)):
        Image.fromarray(np.full((8, 12, 3), value, dtype=np.uint8)).save(directory / name / "000005_10.png")
    Image.fromarray(np.full((8, 12), 4 * 256, dtype=np.uint16)).save(directory / "disp_occ_0" / "000005_10.png")
    sample = KITTI(root=str(tmp_path), image_set="2015")[0]
    torch.testing.assert_close(sample["img1"], torch.full((3, 8, 12), 10.0))
    torch.testing.assert_close(sample["img2"], torch.full((3, 8, 12), 30.0))
    torch.testing.assert_close(sample["disp"], torch.full((8, 12), 4.0))
    assert sample["valid"].all()


def test_custom_validation_subset_is_removed(kitti_root):
    with pytest.raises(ValueError, match="Unknown KITTI image set"):
        KITTI(root=str(kitti_root), image_set="kitti_2015_val")
    with pytest.raises(ValueError, match="Unrecognized dataset"):
        build_val_loader(get_cfg(), "kitti_2015_val")
    root = Path(__file__).resolve().parents[1]
    cfg = get_cfg()
    cfg.merge_from_file(str(root / "configs/kitti_mix_train.yaml"))
    assert list(cfg.DATASETS.TEST) == ["kitti_2015_trainval"]
