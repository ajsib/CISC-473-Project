# src/s2_align/s2b.py

"""
S2B — One-time bounding-box transform into aligned 256×256 space.

Reads:
    results/outputs/s1-validated-pruned-dataset/list_bbox_celeba.csv
    results/outputs/s1-validated-pruned-dataset/list_landmarks_align_celeba.csv

Writes:
    results/outputs/s2-processed-size-bb/list_bbox_celeba.csv

Usage:
    python src/s2_align/s2b.py
"""

import os
import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.s2_align.s2a import (
    CANONICAL_5PT,
    TARGET_SIZE,
    estimate_similarity_transform,
)

logger = get_logger("S2B")


# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------
PRUNED_ROOT = os.path.join("results", "outputs", "s1-validated-pruned-dataset")
S2_OUT_ROOT = os.path.join("results", "outputs", "s2-processed-size-bb")

RAW_BBOX_CSV = os.path.join(PRUNED_ROOT, "list_bbox_celeba.csv")
RAW_LANDMARKS_CSV = os.path.join(PRUNED_ROOT, "list_landmarks_align_celeba.csv")

OUT_BBOX_CSV = os.path.join(S2_OUT_ROOT, "list_bbox_celeba.csv")


# ------------------------------------------------------------
# Bounding box transform helper
# ------------------------------------------------------------
def _transform_bbox(M: np.ndarray, x: float, y: float, w: float, h: float):
    """Transform bbox using affine matrix M."""
    corners = np.array(
        [
            [x, y],
            [x + w, y + h],
        ],
        dtype=np.float32,
    )
    ones = np.ones((2, 1), dtype=np.float32)
    homog = np.hstack([corners, ones])  # shape (2, 3)

    transformed = (M @ homog.T).T  # shape (2, 2)
    xs = transformed[:, 0]
    ys = transformed[:, 1]

    x_min = float(xs.min())
    y_min = float(ys.min())
    x_max = float(xs.max())
    y_max = float(ys.max())

    w_new = x_max - x_min
    h_new = y_max - y_min

    # Clip to 256×256 canvas
    x_min = max(0.0, min(x_min, TARGET_SIZE - 1.0))
    y_min = max(0.0, min(y_min, TARGET_SIZE - 1.0))
    w_new = max(1.0, min(w_new, TARGET_SIZE - x_min))
    h_new = max(1.0, min(h_new, TARGET_SIZE - y_min))

    return x_min, y_min, w_new, h_new


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_s2b():
    logger.info("S2B: Starting bbox transform → aligned 256×256 space.")

    if not os.path.isfile(RAW_BBOX_CSV):
        logger.error("S2B: Missing bbox CSV: %s", RAW_BBOX_CSV)
        raise SystemExit(1)

    if not os.path.isfile(RAW_LANDMARKS_CSV):
        logger.error("S2B: Missing landmarks CSV: %s", RAW_LANDMARKS_CSV)
        raise SystemExit(1)

    # Ensure S2 output directory exists
    os.makedirs(S2_OUT_ROOT, exist_ok=True)

    # Load bbox CSV
    logger.info("S2B: Loading raw bbox CSV: %s", RAW_BBOX_CSV)
    df_bbox = pd.read_csv(
        RAW_BBOX_CSV,
        sep=r"\s+|,",
        engine="python",
        comment="#",
    )

    # Load landmarks CSV
    logger.info("S2B: Loading landmarks: %s", RAW_LANDMARKS_CSV)
    df_landmarks = pd.read_csv(
        RAW_LANDMARKS_CSV,
        sep=r"\s+|,",
        engine="python",
        comment="#",
    )

    # Index landmarks by filename
    lm_id_col = df_landmarks.columns[0]
    df_landmarks = df_landmarks.set_index(lm_id_col)

    bbox_id_col = df_bbox.columns[0]
    bbox_cols = list(df_bbox.columns[1:5])  # x, y, w, h

    n_total = len(df_bbox)
    n_ok = 0
    n_fail = 0

    for idx, row in df_bbox.iterrows():
        fname = str(row[bbox_id_col])

        # Raw bbox values
        try:
            x_raw = float(row[bbox_cols[0]])
            y_raw = float(row[bbox_cols[1]])
            w_raw = float(row[bbox_cols[2]])
            h_raw = float(row[bbox_cols[3]])
        except Exception as e:
            logger.error("S2B: Invalid bbox for '%s': %s", fname, e)
            n_fail += 1
            continue

        # Fetch corresponding landmarks
        try:
            lm_vals = df_landmarks.loc[fname].values.astype(float)
        except KeyError:
            logger.error("S2B: Missing landmarks for '%s'.", fname)
            n_fail += 1
            continue

        if lm_vals.shape[0] != 10:
            logger.error("S2B: Landmark shape invalid for '%s'.", fname)
            n_fail += 1
            continue

        pts = lm_vals.reshape(5, 2)

        # Recompute similarity transform
        M = estimate_similarity_transform(
            src_pts=pts.astype(np.float32),
            dst_pts=CANONICAL_5PT,
        )
        if M is None:
            logger.error("S2B: Transform failed for '%s'.", fname)
            n_fail += 1
            continue

        # Transform bbox → aligned space
        x_al, y_al, w_al, h_al = _transform_bbox(M, x_raw, y_raw, w_raw, h_raw)

        # Store updated aligned-space bbox
        df_bbox.at[idx, bbox_cols[0]] = int(round(x_al))
        df_bbox.at[idx, bbox_cols[1]] = int(round(y_al))
        df_bbox.at[idx, bbox_cols[2]] = int(round(w_al))
        df_bbox.at[idx, bbox_cols[3]] = int(round(h_al))

        n_ok += 1

        if (idx + 1) % 10000 == 0:
            logger.info("S2B: %d / %d processed...", idx + 1, n_total)

    # Write updated bbox CSV into S2 output directory
    df_bbox.to_csv(OUT_BBOX_CSV, index=False)

    logger.info("S2B: Completed. OK=%d FAIL=%d TOTAL=%d", n_ok, n_fail, n_total)
    logger.info("S2B: Output written: %s", OUT_BBOX_CSV)


if __name__ == "__main__":
    run_s2b()
