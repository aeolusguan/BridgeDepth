"""Small readers for the original SceneFlow and Dynamic Replica releases."""

from pathlib import Path

import numpy as np
from PIL import Image


def read_pfm(path):
    """Read bottom-up PFM data, respecting byte order and scale magnitude."""
    with Path(path).open("rb") as stream:
        def header_line():
            while True:
                line = stream.readline()
                if not line:
                    raise ValueError(f"Incomplete PFM header: {path}")
                line = line.strip()
                if line and not line.startswith(b"#"):
                    return line

        magic = header_line()
        if magic not in (b"Pf", b"PF"):
            raise ValueError(f"Invalid PFM magic in {path}: {magic!r}")
        width, height = map(int, header_line().split())
        scale = float(header_line())
        if width <= 0 or height <= 0 or not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Invalid PFM dimensions or scale: {path}")
        channels = 1 if magic == b"Pf" else 3
        values = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
        if values.size != width * height * channels:
            raise ValueError(f"PFM payload size disagrees with header: {path}")
        shape = (height, width) if channels == 1 else (height, width, channels)
        return (np.flipud(values.reshape(shape)) * abs(scale)).astype(np.float32)


def read_float16_depth_png(path):
    """Decode uint16 PNG samples as IEEE float16 *bits*, not integer depth."""
    with Image.open(path) as image:
        if image.mode not in ("I", "I;16", "I;16L", "I;16B"):
            raise ValueError(f"Expected a 16-bit Dynamic Replica depth PNG: {path}")
        bits = np.asarray(image, dtype=np.uint16)
    if bits.ndim != 2:
        raise ValueError(f"Expected a single-channel depth image: {path}")
    return bits.view(np.float16).astype(np.float32)


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def read_sceneflow_poses(path):
    """Parse Frame/L/R records; return OpenCV camera-to-world transforms.

    SceneFlow stores row-major Blender camera-to-world matrices. Only the
    camera axes change here (x right, y down, z forward); world axes stay as
    published. Translation and depth are in Blender units, not metres.
    """
    records = {}
    frame_id = None
    axis_change = np.diag([1.0, -1.0, -1.0, 1.0])
    with Path(path).open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0].lower() == "frame":
                frame_id = int(fields[1])
                records[frame_id] = {}
            elif fields[0] in ("L", "R"):
                if frame_id is None or len(fields) != 17:
                    raise ValueError(f"Malformed camera matrix in {path}: {line.strip()}")
                matrix = np.asarray(fields[1:], dtype=np.float64).reshape(4, 4)
                if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
                    raise ValueError(f"Invalid camera transform in {path}")
                records[frame_id][fields[0]] = (matrix @ axis_change).astype(np.float32)
            else:
                raise ValueError(f"Unrecognized camera record in {path}: {line.strip()}")
    return records
