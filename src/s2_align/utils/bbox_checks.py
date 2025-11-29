import os
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger("S2_BBOX")


def _extract_bbox(df, fname):
    """Extract bounding-box row for filename."""
    id_col = df.columns[0]
    row = df[df[id_col].astype(str) == str(fname)]
    if row.empty:
        return None
    # Bbox columns: x, y, w, h
    vals = row.values[0][1:]
    return tuple(int(v) for v in vals)


def check_bboxes(image_dir, samples, df_bbox, expected_w, expected_h):
    """Check bbox geometry relative to aligned image bounds and expected dimensions."""
    bad = 0

    for fname in samples:
        bbox = _extract_bbox(df_bbox, fname)
        if bbox is None:
            logger.error("S2: Missing bbox row for '%s'.", fname)
            return False

        x, y, w, h = bbox

        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error("S2: Failed to read '%s' for bbox check: %s", fname, e)
            return False

        W, H = img.size

        if W != expected_w or H != expected_h:
            bad += 1
            if bad <= 10:
                logger.error(
                    "S2: Image '%s' size %dx%d does not match expected %dx%d.",
                    fname, W, H, expected_w, expected_h
                )
            continue

        if not (0 <= x < W and 0 <= y < H):
            bad += 1
            if bad <= 10:
                logger.error("S2: Bbox origin outside image for '%s': %s", fname, bbox)
            continue

        if x + w > W or y + h > H:
            bad += 1
            if bad <= 10:
                logger.error("S2: Bbox extends outside image for '%s': %s", fname, bbox)
            continue

    if bad > 0:
        logger.error("S2: %d sampled images have invalid bounding boxes.", bad)
        return False

    logger.info("S2: All sampled bounding boxes are inside image bounds.")
    return True
