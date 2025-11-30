import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger("S4B_MODEL")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------


def _project_root() -> Path:
    """
    project/
      src/
        s4_codeformer/
          utils/
            model.py   <-- this file
    -> parents[3] is project root.
    """
    return Path(__file__).resolve().parents[3]


def _codeformer_repo() -> Path:
    root = _project_root()
    repo = root / "CodeFormer"
    if not repo.is_dir():
        raise FileNotFoundError(
            f"S4B: CodeFormer repo not found at {repo}. "
            "Run environment wizard or clone the repo under project/CodeFormer."
        )
    return repo


# ---------------------------------------------------------------------
# Weights resolution
# ---------------------------------------------------------------------


def _candidate_weight_paths(ckpt_name: str) -> List[Path]:
    """
    Search order:
      1. $CODEFORMER_WEIGHTS/ckpt_name
      2. project/weights/codeformer/ckpt_name
      3. project/CodeFormer/weights/CodeFormer/ckpt_name  (repo default)
      4. project/.cache/codeformer/ckpt_name
      5. ~/.cache/codeformer/ckpt_name
    """
    fn = ckpt_name if ckpt_name.endswith(".pth") else f"{ckpt_name}.pth"

    env_dir = os.environ.get("CODEFORMER_WEIGHTS")
    paths: List[Path] = []

    if env_dir:
        paths.append(Path(env_dir) / fn)

    root = _project_root()
    repo = _codeformer_repo()

    paths.extend(
        [
            root / "weights" / "codeformer" / fn,
            repo / "weights" / "CodeFormer" / fn,
            root / ".cache" / "codeformer" / fn,
            Path.home() / ".cache" / "codeformer" / fn,
        ]
    )
    return paths


def _resolve_weights(ckpt_name: str) -> Path:
    for p in _candidate_weight_paths(ckpt_name):
        if p.is_file():
            logger.info("S4B: Using CodeFormer weights '%s' from %s", ckpt_name, p)
            return p

    msg = (
        f"CodeFormer weights not found: '{ckpt_name}'. "
        f"Expected under either:\n"
        f"  - {(_project_root() / 'weights' / 'codeformer')}\n"
        f"  - {(_codeformer_repo() / 'weights' / 'CodeFormer')}\n"
        f"or set CODEFORMER_WEIGHTS=/path/to/dir"
    )
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------
# basicsr import (CodeFormer-local, not pip)
# ---------------------------------------------------------------------


def _import_codeformer_basicsr() -> None:
    """
    Hard-bind the CodeFormer-local basicsr as the *only* 'basicsr' package
    visible for this stage.

    This completely ignores / replaces any pip-installed 'basicsr', so S4A
    (GFPGAN) can safely use its own pip basicsr in a separate run.
    """
    repo = _codeformer_repo()
    basicsr_dir = repo / "basicsr"
    init_py = basicsr_dir / "__init__.py"

    if not init_py.is_file():
        raise FileNotFoundError(
            f"S4B: basicsr/__init__.py not found under {basicsr_dir} "
            "(CodeFormer repo looks incomplete)."
        )

    # Ensure CodeFormer root is importable for 'facelib', etc.
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # Drop any previously-loaded 'basicsr' (pip or otherwise)
    for name in list(sys.modules.keys()):
        if name == "basicsr" or name.startswith("basicsr."):
            sys.modules.pop(name, None)

    # Load basicsr from CodeFormer into sys.modules["basicsr"]
    spec = importlib.util.spec_from_file_location(
        "basicsr",
        init_py,
        submodule_search_locations=[str(basicsr_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["basicsr"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    logger.info("S4B: Using CodeFormer basicsr from %s", basicsr_dir)


def _try_imports() -> bool:
    """
    Return True if we can import torch + CodeFormer from the CodeFormer-local
    basicsr. False otherwise (and log a clear error).
    """
    try:
        _import_codeformer_basicsr()
    except Exception as e:
        logger.error("S4B: Failed to bind CodeFormer basicsr: %s", e)
        return False

    try:
        import torch  # noqa: F401

        from basicsr.archs.codeformer_arch import CodeFormer  # noqa: F401

        logger.info("S4B: torch + CodeFormer imports OK.")
        return True
    except Exception as e:
        logger.error(
            "S4B: CodeFormer import failed (basicsr.archs.codeformer_arch): %s",
            e,
        )
        return False


# ---------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------


def _get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def load_codeformer(
    ckpt_name: str = "codeformer-v0.1.0",
) -> Optional[Dict[str, Any]]:
    """
    Returns a dict with:
      {
        "net": CodeFormer,
        "device": "cuda" | "cpu"
      }
    or None on failure.
    """
    if not _try_imports():
        return None

    import torch
    from basicsr.archs.codeformer_arch import CodeFormer

    model_path = _resolve_weights(ckpt_name)
    device = _get_device()

    logger.info("S4B: Building CodeFormer model on device '%s'.", device)

    net = CodeFormer(
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
    )

    ckpt = torch.load(model_path, map_location=device)

    if "params_ema" in ckpt:
        state = ckpt["params_ema"]
    elif "state_dict" in ckpt:
        state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    else:
        raise KeyError(
            "S4B: Unexpected CodeFormer checkpoint format "
            "(expected 'params_ema' or 'state_dict')."
        )

    net.load_state_dict(state, strict=True)
    net.eval()
    net.to(device)

    logger.info("S4B: CodeFormer weights loaded and model ready.")
    return {"net": net, "device": device}


def run_codeformer(
    model: Dict[str, Any],
    img_pil: Image.Image,
    fidelity: float,
) -> Image.Image:
    """
    Run CodeFormer on a single PIL RGB image with given fidelity w in [0,1].
    Returns a PIL RGB image.
    """
    import torch
    from torchvision import transforms

    net = model["net"]
    device = model["device"]

    to_tensor = transforms.ToTensor()
    x = to_tensor(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(x, w=fidelity)[0]

    out = out.clamp(0, 1).cpu().numpy()
    out = (out.transpose(1, 2, 0) * 255).astype(np.uint8)

    return Image.fromarray(out, "RGB")
