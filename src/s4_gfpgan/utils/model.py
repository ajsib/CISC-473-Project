import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger("S4A_MODEL")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_weight_paths(ckpt_name: str) -> List[Path]:
    fn = ckpt_name if ckpt_name.endswith(".pth") else f"{ckpt_name}.pth"
    env_dir = os.environ.get("GFPGAN_WEIGHTS")
    paths: List[Path] = []
    if env_dir:
        paths.append(Path(env_dir) / fn)
    paths.extend(
        [
            _project_root() / "weights" / "gfpgan" / fn,
            _project_root() / ".cache" / "gfpgan" / fn,
            Path.home() / ".cache" / "gfpgan" / fn,
        ]
    )
    return paths


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _try_download(ckpt_name: str) -> Optional[Path]:
    ckpt_file = ckpt_name if ckpt_name.endswith(".pth") else f"{ckpt_name}.pth"
    url = f"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/{ckpt_file}"
    target = Path.home() / ".cache" / "gfpgan" / ckpt_file
    try:
        import requests

        _ensure_parent(target)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return target
    except Exception as e:
        logger.warning("S4A: Download of GFPGAN weights failed: %s", e)
        return None


def _resolve_weights(ckpt_name: str) -> Path:
    for p in _candidate_weight_paths(ckpt_name):
        if p.is_file():
            return p

    dl = _try_download(ckpt_name)
    if dl and dl.is_file():
        return dl

    hint = _project_root() / "weights" / "gfpgan" / (
        ckpt_name if ckpt_name.endswith(".pth") else f"{ckpt_name}.pth"
    )
    raise FileNotFoundError(
        f"GFPGAN weights not found: '{ckpt_name}'. "
        f"Place file at '{hint}' or set GFPGAN_WEIGHTS=/path/to/dir"
    )


# Delayed imports so the CLI still loads even if deps are missing.
def _try_imports() -> bool:
    try:
        import torch  # noqa
        from gfpgan import GFPGANer  # noqa
        import cv2  # noqa

        return True
    except Exception as e:
        logger.error(
            "S4A: Missing GFPGAN runtime deps. Install (pinned) packages:\n"
            "  pip install 'numpy<2' gfpgan basicsr facexlib realesrgan opencv-python\n"
            "Error: %s",
            e,
        )
        return False


def get_device_str() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_gfpgan(ckpt_name: str = "GFPGANv1.4", upscale: int = 1):
    """
    Returns a configured GFPGANer instance or None on failure.
    For aligned 256×256 inputs, use has_aligned=True at inference.
    """
    if not _try_imports():
        return None

    from gfpgan import GFPGANer

    model_path = _resolve_weights(ckpt_name)
    restorer = GFPGANer(
        model_path=str(model_path),
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=get_device_str(),
    )
    logger.info(
        "S4A: Loaded GFPGAN weights '%s' from %s on device '%s' (upscale=%d).",
        ckpt_name,
        model_path,
        restorer.device,
        upscale,
    )
    return restorer


def enhance_aligned_pil(
    restorer, img: Image.Image, enforce_size: Optional[Union[int, Tuple[int, int]]] = 256
) -> Image.Image:
    """
    Run GFPGAN on a single aligned face image (PIL, RGB), return PIL RGB.
    GFPGAN expects BGR np.uint8; we use has_aligned=True to get restored_face.

    Adds strict input validation so OpenCV never sees an empty array.
    """
    import cv2

    # PIL -> numpy (RGB)
    arr_rgb = np.array(img, copy=False)

    if arr_rgb is None or arr_rgb.size == 0:
        raise RuntimeError("S4A: Empty image array passed to GFPGAN (decode failure).")

    if arr_rgb.ndim != 3 or arr_rgb.shape[2] not in (3, 4):
        raise RuntimeError(
            f"S4A: Unexpected image array shape {arr_rgb.shape}; expected HxWx3 or HxWx4."
        )

    # Drop alpha if present
    if arr_rgb.shape[2] == 4:
        arr_rgb = arr_rgb[:, :, :3]

    # RGB -> BGR
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

    # GFPGAN inference
    _, _, restored = restorer.enhance(
        arr_bgr,
        has_aligned=False,
        only_center_face=True,
        paste_back=False,
    )

    if restored is None or restored.size == 0:
        raise RuntimeError("S4A: GFPGAN returned an empty restored frame.")

    out_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(out_rgb).convert("RGB")

    if enforce_size is not None:
        if isinstance(enforce_size, int):
            target = (enforce_size, enforce_size)
        else:
            target = enforce_size
        if out.size != target:
            out = out.resize(target, Image.BICUBIC)
    return out
