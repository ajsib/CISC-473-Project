# src/s1_data/utils/fs.py

import os
from typing import Dict, Iterable, Set

import pandas as pd

from src.utils.logging import get_logger


logger = get_logger("S1_FS")


def ensure_required_paths(image_dir: str, csv_paths: Dict[str, str]) -> None:
    """Ensure that the image directory and all metadata CSVs exist.

    Raises SystemExit on any missing path.
    """
    if not os.path.isdir(image_dir):
        logger.error("S1: Expected image directory '%s' not found.", image_dir)
        raise SystemExit(1)

    for name, path in csv_paths.items():
        if not os.path.isfile(path):
            logger.error("S1: Expected CSV '%s' for '%s' not found.", path, name)
            raise SystemExit(1)

    logger.info("S1: All required dataset paths are present.")


def list_image_filenames(image_dir: str) -> Set[str]:
    """Return the set of image filenames (with extensions) under image_dir."""
    valid_ext = {".jpg", ".jpeg", ".png"}
    images: Set[str] = set()

    try:
        for fname in os.listdir(image_dir):
            _, ext = os.path.splitext(fname)
            if ext.lower() in valid_ext:
                images.add(fname)
    except OSError as e:
        logger.error("S1: Failed to list images in '%s': %s", image_dir, e)
        raise SystemExit(1)

    if images:
        sample = sorted(list(images))[:5]
        logger.info("S1: Sample image filenames: %s", ", ".join(sample))

    return images


def _read_generic_csv(path: str) -> pd.DataFrame:
    """Read a CelebA-style CSV with flexible parsing.

    Uses a regex separator to handle both whitespace- and comma-separated files.
    """
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+|,",
            engine="python",
            comment="#",
        )
    except Exception as e:
        logger.error("S1: Failed to read CSV '%s': %s", path, e)
        raise SystemExit(1)

    if df.empty:
        logger.error("S1: CSV '%s' is empty.", path)
        raise SystemExit(1)

    return df


def load_metadata_frames(csv_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Load all metadata CSVs into DataFrames.

    Keys in csv_paths map to the same keys in the returned dict.
    """
    frames: Dict[str, pd.DataFrame] = {}
    for key, path in csv_paths.items():
        df = _read_generic_csv(path)
        logger.info(
            "S1: Loaded CSV '%s' (%s) with %d rows and %d columns.",
            path,
            key,
            len(df),
            df.shape[1],
        )
        frames[key] = df

    return frames
