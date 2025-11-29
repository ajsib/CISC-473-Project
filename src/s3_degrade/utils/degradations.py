import io
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageFilter

from src.utils.logging import get_logger

logger = get_logger("S3_DEG")


def apply_degradation(img: Image.Image, preset: Dict, output_size: Optional[int]) -> Image.Image:
    """
    Apply a single degradation preset to a PIL image and optionally enforce output_size×output_size.

    Supported preset["type"]:
        - "gaussian_blur": uses preset["sigma"] as blur radius
        - "jpeg": uses preset["quality"] as JPEG quality
        - "gaussian_noise": uses preset["sigma"] as pixel-wise stddev in [0,255] space
    """
    preset_type = preset.get("type")
    name = preset.get("name", "<unnamed>")

    if preset_type == "gaussian_blur":
        sigma = float(preset["sigma"])
        img_out = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    elif preset_type == "jpeg":
        quality = int(preset["quality"])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        img_out = Image.open(buf).convert("RGB")

    elif preset_type == "gaussian_noise":
        sigma = float(preset["sigma"])
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
        arr_noisy = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        img_out = Image.fromarray(arr_noisy)

    else:
        raise ValueError(f"S3: Unsupported degradation type '{preset_type}' for preset '{name}'")

    if output_size is not None and img_out.size != (output_size, output_size):
        logger.warning(
            "S3: Image size %s after degradation '%s' does not match target %dx%d. Resizing.",
            img_out.size, name, output_size, output_size,
        )
        img_out = img_out.resize((output_size, output_size), Image.BICUBIC)

    return img_out
