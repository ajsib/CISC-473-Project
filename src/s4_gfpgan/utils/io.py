import os
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.utils.logging import get_logger

logger = get_logger("S4A_IO")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_manifest_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"S4A: S3 manifest not found: {path}")
    df = pd.read_csv(path)
    required = {"id", "path_gt", "path_deg", "degradation", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"S4A: Manifest missing columns: {sorted(list(missing))}")
    return df


def load_image_rgb(path: str) -> Image.Image:
    """
    Robust image loader for S4A.

    Guarantees:
    - Path exists and is a regular file.
    - File size is non-trivial (> 0 bytes).
    - PIL fully decodes the image (img.load()).
    - Image is converted to RGB.
    - Image has positive width/height.

    Raises RuntimeError with a clear message if anything goes wrong.
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"S4A: Missing input image file: {path}")

    size_bytes = os.path.getsize(path)
    if size_bytes <= 0:
        raise RuntimeError(f"S4A: Zero-byte input image: {path}")

    try:
        with Image.open(path) as im:
            im.load()  # force full decode
            img = im.convert("RGB")
    except UnidentifiedImageError as e:
        raise RuntimeError(f"S4A: Unidentified/unsupported image file: {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"S4A: Failed to decode image: {path}: {e}") from e

    w, h = img.size
    if w <= 0 or h <= 0:
        raise RuntimeError(
            f"S4A: Decoded image has invalid size {w}x{h}: {path}"
        )

    return img


def save_image_jpeg(img: Image.Image, path: str, quality: int = 95) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="JPEG", quality=quality, optimize=True)


def count_existing(paths: Iterable[str]) -> int:
    return sum(1 for p in paths if os.path.isfile(p))
