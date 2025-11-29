# src/s3_degrade/utils/io.py

import os
from typing import Iterable, List, Optional, Tuple

from PIL import Image

from src.utils.logging import get_logger

logger = get_logger("S3_IO")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_aligned_filenames(root: str) -> List[str]:
    # Assumes files are directly under root; adapt if nested.
    return sorted([f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))])


def load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _is_valid_rgb(path: str, expect_size: Optional[Tuple[int, int]] = None) -> bool:
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 1024:
            return False
        with Image.open(path) as im:
            im.load()
            if im.mode not in ("RGB", "RGBA", "L"):
                im.convert("RGB")
            w, h = im.size
            if w <= 0 or h <= 0:
                return False
            if expect_size is not None and (w, h) != expect_size:
                return False
        return True
    except Exception:
        return False


def list_valid_rgb_images(root: str, expect_size: Optional[Tuple[int, int]] = None) -> List[str]:
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for f in os.listdir(root):
        p = os.path.join(root, f)
        if os.path.isfile(p) and _is_valid_rgb(p, expect_size=expect_size):
            out.append(f)
    return out


def verify_rgb_images_ok(paths: Iterable[str], expect_size: Optional[Tuple[int, int]] = None) -> int:
    ok = 0
    for p in paths:
        if _is_valid_rgb(p, expect_size=expect_size):
            ok += 1
    return ok


def save_image_atomic(img: Image.Image, path: str, quality: int = 95) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    img = img.convert("RGB")
    img.save(tmp, format="JPEG", quality=quality, optimize=True)
    os.replace(tmp, path)
