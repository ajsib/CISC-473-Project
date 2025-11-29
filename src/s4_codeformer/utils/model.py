import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_weight_paths(ckpt_name: str):
    fn = ckpt_name if ckpt_name.endswith(".pth") else f"{ckpt_name}.pth"
    env = os.environ.get("CODEFORMER_WEIGHTS")
    paths = []
    if env:
        paths.append(Path(env) / fn)
    paths.extend(
        [
            _project_root() / "weights" / "codeformer" / fn,
            _project_root() / ".cache" / "codeformer" / fn,
            Path.home() / ".cache" / "codeformer" / fn,
        ]
    )
    return paths


def _resolve_weights(ckpt_name: str) -> Path:
    for p in _candidate_weight_paths(ckpt_name):
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"CodeFormer weights not found: '{ckpt_name}'. "
        "Place file under weights/codeformer/ or set CODEFORMER_WEIGHTS=/path"
    )


def _ensure_repo_present():
    root = _project_root()
    repo = root / "CodeFormer"

    print("S4B-DIAG: Checking CodeFormer repo at:", repo)

    if repo.exists():
        print("S4B-DIAG: CodeFormer repo exists")
        return

    print("S4B-DIAG: CodeFormer repo missing. Cloning...")
    url = "https://github.com/sczhou/CodeFormer.git"

    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(repo)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("S4B-DIAG: CodeFormer cloned successfully.")

def _inject_repo():
    import sys

    _ensure_repo_present()

    root = _project_root()
    repo = root / "CodeFormer"
    basicsr_dir = repo / "basicsr"

    print("S4B-DIAG: project root =", root)
    print("S4B-DIAG: repo =", repo)
    print("S4B-DIAG: basicsr_dir =", basicsr_dir)

    # add BOTH paths
    sys.path.insert(0, str(basicsr_dir))
    sys.path.insert(0, str(repo))

    print("S4B-DIAG: sys.path[0:5] =", sys.path[:5])



def _try_imports() -> bool:
    try:
        print("S4B-DIAG: injecting CodeFormer repo...")
        _inject_repo()
    except Exception as e:
        print("S4B-DIAG: repo injection failed:", repr(e))
        return False

    try:
        import torch
        print("S4B-DIAG: torch import OK")
    except Exception as e:
        print("S4B-DIAG: torch import FAILED:", repr(e))
        return False

    try:
        from basicsr.archs.codeformer_arch import CodeFormer
        print("S4B-DIAG: CodeFormer import OK (basicsr.archs.codeformer_arch)")
    except Exception as e:
        print("S4B-DIAG: CodeFormer import FAILED:", repr(e))
        return False

    return True



def load_codeformer(ckpt_name="codeformer-v0.1.0"):

    if not _try_imports():
        return None

    import torch
    from projects.CodeFormer.codeformer_arch import CodeFormer

    model_path = _resolve_weights(ckpt_name)
    device = "cpu"

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
        raise KeyError("Unexpected CodeFormer checkpoint format.")

    net.load_state_dict(state, strict=True)
    net.eval()
    net.to(device)

    return {"net": net, "device": device}


def run_codeformer(model, img_pil: Image.Image, fidelity: float):
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
