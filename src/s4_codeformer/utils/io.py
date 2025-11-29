import os
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_manifest_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"S4B: S3 manifest not found: {path}")
    df = pd.read_csv(path)
    required = {"id", "path_gt", "path_deg", "degradation", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"S4B: Manifest missing columns: {sorted(list(missing))}")
    return df


def load_image_rgb(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise RuntimeError(f"S4B: Missing input image file: {path}")

    size_bytes = os.path.getsize(path)
    if size_bytes <= 0:
        raise RuntimeError(f"S4B: Zero-byte input image: {path}")

    try:
        with Image.open(path) as im:
            im.load()
            img = im.convert("RGB")
    except UnidentifiedImageError as e:
        raise RuntimeError(f"S4B: Unidentified image file: {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"S4B: Failed to decode image: {path}: {e}") from e

    w, h = img.size
    if w <= 0 or h <= 0:
        raise RuntimeError(f"S4B: Decoded image has invalid size {w}x{h}: {path}")

    return img


def save_image_jpeg(img: Image.Image, path: str, quality: int = 95) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="JPEG", quality=quality, optimize=True)


def count_existing(paths: Iterable[str]) -> int:
    return sum(1 for p in paths if os.path.isfile(p))
