import os
import numpy as np
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger("S2_LANDMARK")


def _extract_landmarks(df, fname):
    """Extract landmark row for filename. Returns None if missing."""
    id_col = df.columns[0]
    row = df[df[id_col].astype(str) == str(fname)]
    if row.empty:
        return None
    # Landmarks are 10 columns: l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, ...
    values = row.values[0][1:]
    return np.array(values, dtype=float).reshape(-1, 2)


def check_landmarks(image_dir, samples, df_landmarks, expected_w, expected_h):
    """Check that landmarks lie inside image bounds and image dims match expected geometry."""
    bad = 0

    for fname in samples:
        lm = _extract_landmarks(df_landmarks, fname)
        if lm is None:
            logger.error("S2: Missing landmarks row for '%s'.", fname)
            return False

        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error("S2: Failed to read '%s' for landmark check: %s", fname, e)
            return False

        w, h = img.size

        if w != expected_w or h != expected_h:
            bad += 1
            if bad <= 10:
                logger.error(
                    "S2: Image '%s' size %dx%d does not match expected %dx%d.",
                    fname, w, h, expected_w, expected_h
                )
            continue

        if not ((lm[:, 0] >= 0).all() and (lm[:, 0] < w).all() and
                (lm[:, 1] >= 0).all() and (lm[:, 1] < h).all()):
            bad += 1
            if bad <= 10:
                logger.error("S2: Landmarks for '%s' fall outside image bounds.", fname)

    if bad > 0:
        logger.error("S2: %d sampled images have out-of-bounds landmarks.", bad)
        return False

    logger.info("S2: All sampled landmarks lie within image bounds.")
    return True
