import os
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger("S2_IMAGE")


def check_image_geometry(image_dir, samples, expected_w, expected_h):
    """Verify that sampled aligned images satisfy expected geometry:
    - width matches expected_w
    - height matches expected_h
    - exactly 3 channels
    """
    ok = True
    bad_samples = []

    for fname in samples:
        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path)
        except Exception as e:
            logger.error("S2: Failed to read '%s': %s", path, e)
            return False

        w, h = img.size
        channels = len(img.getbands())

        if w != expected_w or h != expected_h or channels != 3:
            ok = False
            bad_samples.append(fname)
            if len(bad_samples) <= 10:
                logger.error(
                    "S2: Image '%s' is %dx%d with %d channels (expected %dx%d, 3 channels).",
                    fname, w, h, channels, expected_w, expected_h
                )

    if not ok:
        logger.error(
            "S2: %d images do not satisfy required geometry.",
            len(bad_samples),
        )
        return False

    logger.info(
        "S2: All sampled images match expected geometry %dx%d with 3 channels.",
        expected_w, expected_h
    )
    return True

    logger.info(
        "S2: All sampled images match expected geometry %dx%d with 3 channels.",
        expected_w, expected_h
    )
    return True
