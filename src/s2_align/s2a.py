# src/s2_align/s2a.py

"""
S2A — One-time CelebA alignment and canonical 256×256 resizing.

Reads validated CelebA images from:
    results/outputs/s1-validated-pruned-dataset/img_align_celeba/

Writes aligned 256×256 images to:
    results/outputs/s2-processed-size-bb/img_align_celeba/

Usage (developer only, run once):
    python src/s2_align/s2a.py
"""

import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("S2A")


# ------------------------------------------------------------
# Canonical target 5-point geometry for 256×256 alignment
# ------------------------------------------------------------
CANONICAL_5PT = np.array([
    [70.0,  100.0],   # left eye
    [186.0, 100.0],   # right eye
    [128.0, 142.0],   # nose
    [88.0,  182.0],   # left mouth
    [168.0, 182.0],   # right mouth
], dtype=np.float32)

TARGET_SIZE = 256


# ------------------------------------------------------------
# Directories (as required)
# ------------------------------------------------------------
PRUNED_ROOT = os.path.join("results", "outputs", "s1-validated-pruned-dataset")
S2_OUT_ROOT = os.path.join("results", "outputs", "s2-processed-size-bb")

RAW_IMG_DIR = os.path.join(PRUNED_ROOT, "img_align_celeba")
OUT_IMG_DIR = os.path.join(S2_OUT_ROOT, "img_align_celeba")

LANDMARK_CSV = os.path.join(PRUNED_ROOT, "list_landmarks_align_celeba.csv")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_landmarks(df, fname):
    """Return 5×2 landmarks for a given filename."""
    id_col = df.columns[0]
    row = df[df[id_col].astype(str) == str(fname)]
    if row.empty:
        return None
    vals = row.values[0][1:]
    return np.array(vals, dtype=float).reshape(5, 2)


def estimate_similarity_transform(src_pts, dst_pts):
    """Compute similarity transform."""
    M = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)[0]
    return M


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_s2a():
    logger.info("S2A: Starting CelebA → canonical 256×256 alignment.")

    if not os.path.isdir(RAW_IMG_DIR):
        logger.error("S2A: Missing raw image directory: %s", RAW_IMG_DIR)
        raise SystemExit(1)

    if not os.path.isfile(LANDMARK_CSV):
        logger.error("S2A: Missing landmark CSV: %s", LANDMARK_CSV)
        raise SystemExit(1)

    ensure_dir(S2_OUT_ROOT)
    ensure_dir(OUT_IMG_DIR)

    # Load landmark dataframe
    df_landmarks = pd.read_csv(LANDMARK_CSV)
    logger.info("S2A: Loaded %d landmark rows.", len(df_landmarks))

    # List images in pruned dataset
    raw_fnames = sorted([f for f in os.listdir(RAW_IMG_DIR) if f.endswith(".jpg")])
    logger.info("S2A: Found %d images for alignment.", len(raw_fnames))

    processed = 0
    skipped = 0

    for fname in raw_fnames:
        in_path = os.path.join(RAW_IMG_DIR, fname)

        # Read image
        try:
            img = Image.open(in_path).convert("RGB")
        except Exception as e:
            logger.error("S2A: Cannot read '%s': %s", fname, e)
            skipped += 1
            continue

        raw_arr = np.array(img)
        h, w = raw_arr.shape[:2]

        # Read landmarks
        pts = read_landmarks(df_landmarks, fname)
        if pts is None:
            logger.error("S2A: Missing landmarks for '%s'.", fname)
            skipped += 1
            continue

        # Validate bounds
        if not ((pts[:, 0] >= 0).all() and (pts[:, 0] < w).all() and
                (pts[:, 1] >= 0).all() and (pts[:, 1] < h).all()):
            logger.error("S2A: Out-of-bound landmarks in '%s'.", fname)
            skipped += 1
            continue

        # Compute transform
        M = estimate_similarity_transform(
            src_pts=pts.astype(np.float32),
            dst_pts=CANONICAL_5PT,
        )
        if M is None:
            logger.error("S2A: Transform failed for '%s'.", fname)
            skipped += 1
            continue

        # Apply warp
        aligned = cv2.warpAffine(
            raw_arr,
            M,
            (TARGET_SIZE, TARGET_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Save
        out_path = os.path.join(OUT_IMG_DIR, fname)
        Image.fromarray(aligned).save(out_path, quality=95)

        processed += 1
        if processed % 5000 == 0:
            logger.info("S2A: Processed %d images...", processed)

    logger.info("S2A: Completed. Processed=%d, Skipped=%d", processed, skipped)
    logger.info("S2A: Output directory: %s", OUT_IMG_DIR)


if __name__ == "__main__":
    run_s2a()
