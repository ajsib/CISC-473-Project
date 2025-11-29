import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from typing import Optional

from src.utils.logging import get_logger

LOGGER = get_logger("ENV")

MODULE_NAME_OVERRIDES: Dict[str, str] = {
    "opencv-python": "cv2",
    "PyYAML": "yaml",
    "pillow": "PIL",
    "scikit-image": "skimage",
}


# ---------------------------------------------------------------------------
# Internal paths
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_path() -> Path:
    return _project_root() / "config.json"


def _host_python() -> Path:
    """
    Return a python interpreter outside the target .venv.
    When already inside a venv, prefer the base interpreter to avoid deleting
    the running binary when rebuilding the env.
    """
    if hasattr(sys, "real_prefix"):
        base = Path(sys.real_prefix)
    elif hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
        base = Path(sys.base_prefix)
    else:
        base = None

    candidate = None
    if base:
        candidate = base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    # Fall back to the current interpreter if we can't find a better host.
    return candidate if candidate and candidate.is_file() else Path(sys.executable)


# ---------------------------------------------------------------------------
# Load config: env.{python,dependencies} with fallback to stack.versions.python_packages
# env.dependencies take precedence over stack pins.
# ---------------------------------------------------------------------------

def _load_env_config() -> Tuple[Dict[str, str], Dict[str, str]]:
    cfg_path = _config_path()
    if not cfg_path.is_file():
        LOGGER.error("config.json missing at %s", cfg_path)
        return {}, {}

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    env_cfg = cfg.get("env", {}) or {}
    deps_env = env_cfg.get("dependencies", {}) or {}
    deps_stack = (
        cfg.get("stack", {})
           .get("versions", {})
           .get("python_packages", {}) or {}
    )
    # env dependencies override stack pins
    merged_deps = {**deps_stack, **deps_env}
    return env_cfg.get("python", {}) or {}, merged_deps


# ---------------------------------------------------------------------------
# Version checks
# ---------------------------------------------------------------------------

def _parse_version(v: Optional[str]):
    if not v:
        return None
    parts = v.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def _check_python_version(python_cfg: Dict[str, str]) -> Tuple[bool, str]:
    major, minor = sys.version_info[:2]
    current = (major, minor)
    cur_str = f"{major}.{minor}"

    min_v = _parse_version(python_cfg.get("min"))
    max_v = _parse_version(python_cfg.get("max"))

    if min_v and current < min_v:
        return False, f"Python {cur_str} < min required {min_v[0]}.{min_v[1]}"
    if max_v and current > max_v:
        return False, f"Python {cur_str} > supported max {max_v[0]}.{max_v[1]}"

    return True, f"Python {cur_str} OK"


# ---------------------------------------------------------------------------
# Requirements from config.json
# ---------------------------------------------------------------------------

def _write_requirements(deps: Dict[str, str]) -> Path:
    path = _project_root() / "requirements.txt"
    with path.open("w", encoding="utf-8") as f:
        for pkg, ver in deps.items():
            if ver:
                f.write(f"{pkg}=={ver}\n")
            else:
                f.write(f"{pkg}\n")
    LOGGER.info("requirements.txt written to %s", path)
    return path


# ---------------------------------------------------------------------------
# .venv python helper
# ---------------------------------------------------------------------------

def _venv_python() -> Path:
    root = _project_root()
    return root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")


# ---------------------------------------------------------------------------
# PyTorch index helper
# ---------------------------------------------------------------------------

def _torch_extra_index(deps: Dict[str, str]) -> Optional[str]:
    # tv = (deps.get("torch") or "").strip()
    # if tv.startswith("1.13"):
    #     return "https://download.pytorch.org/whl/cu117"
    # if tv.startswith("2.3"):
    #     return "https://download.pytorch.org/whl/cu121"
    return None


# ---------------------------------------------------------------------------
# Create .venv using the host interpreter
# ---------------------------------------------------------------------------

def _setup_venv() -> Optional[Path]:
    root = _project_root()
    venv_dir = root / ".venv"

    # Never nuke the interpreter we're running on
    inside = sys.prefix != sys.base_prefix
    current_prefix = Path(sys.prefix).resolve()
    if inside and current_prefix == venv_dir.resolve():
        LOGGER.warning("Active venv detected; skipping rebuild.")
        return _venv_python() if _venv_python().exists() else None

    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        LOGGER.info("Removed old .venv")

    # Build using the host/base interpreter (not the current venv)
    base_python = Path(sys.base_prefix) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    builder = base_python if base_python.exists() else Path(sys.executable)

    try:
        subprocess.run([str(builder), "-m", "venv", str(venv_dir)], check=True)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Could not create .venv: %s", e)
        return None

    return _venv_python()


# ---------------------------------------------------------------------------
# Install deps inside venv
# ---------------------------------------------------------------------------

def _install_requirements(python_exec: Path, req_path: Path, extra_index_url: Optional[str] = None):
    try:
        subprocess.run([str(python_exec), "-m", "pip", "install", "--upgrade", "pip"], check=True)

        cmd = [
            str(python_exec), "-m", "pip", "install",
            "--no-cache-dir",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--extra-index-url", "https://pypi.org/simple",
            "-r", str(req_path)
        ]

        # UNCOMMENT FOR CUDA ---V

        # cmd = [str(python_exec), "-m", "pip", "install", "-r", str(req_path), "--no-cache-dir"]
        # if extra_index_url:
        #     cmd += ["--extra-index-url", extra_index_url]

        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed pip install: %s", e)
        print("Failed to install dependencies.")
        return False
    return True


# ---------------------------------------------------------------------------
# Module import checks
# ---------------------------------------------------------------------------

def _map_pkg(pkg: str) -> str:
    return MODULE_NAME_OVERRIDES.get(pkg, pkg.replace("-", "_"))


def _patch_torchvision_alias() -> None:
    """
    gfpgan/basicsr/realesrgan import torchvision.transforms.functional_tensor,
    which was renamed to _functional_tensor in newer torchvision releases.
    Provide a compat alias when needed.
    """
    import importlib

    if "torchvision.transforms.functional_tensor" in sys.modules:
        return

    try:
        importlib.import_module("torchvision.transforms.functional_tensor")
        return
    except ModuleNotFoundError:
        pass

    try:
        impl = importlib.import_module("torchvision.transforms._functional_tensor")
    except Exception:
        return

    sys.modules["torchvision.transforms.functional_tensor"] = impl


def _check_imports(deps: Dict[str, str]) -> Tuple[bool, List[str]]:
    _patch_torchvision_alias()
    msgs = []
    ok = True
    for pkg in deps.keys():
        mod = _map_pkg(pkg)
        try:
            importlib.import_module(mod)
            msgs.append(f"OK - import {mod}")
        except Exception as e:
            ok = False
            msgs.append(f"FAIL - import {mod}: {e}")
    return ok, msgs


def _check_cuda() -> Tuple[bool, List[str]]:
    msgs = []
    try:
        import torch
        msgs.append(f"OK - torch {torch.__version__}")
        if torch.cuda.is_available():
            msgs.append(f"OK - CUDA available ({torch.cuda.device_count()} GPU(s))")
        else:
            msgs.append("WARN - CUDA not available (CPU mode)")
        return True, msgs
    except Exception as e:
        return False, [f"FAIL - torch/CUDA: {e}"]


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def check_environment() -> Tuple[bool, List[str]]:
    lines = [f"Python {sys.version_info[0]}.{sys.version_info[1]} OK",
             f"Interpreter: {sys.executable}"]

    py_cfg, deps = _load_env_config()

    ok_py, msg_py = _check_python_version(py_cfg)
    (LOGGER.info if ok_py else LOGGER.error)(msg_py)

    ok_mod, mod_lines = _check_imports(deps)
    lines.extend(mod_lines)

    ok_cuda, cuda_lines = _check_cuda()
    lines.extend(cuda_lines)

    ok = ok_py and ok_mod and ok_cuda
    lines.append("Environment check PASSED." if ok else "Environment check FAILED.")
    return ok, lines


# ---------------------------------------------------------------------------
# Environment Wizard (single flow)
# ---------------------------------------------------------------------------

def run_environment_wizard() -> None:
    ok, lines = check_environment()

    print("\n--- Environment check ---")
    for line in lines:
        print(line)

    if ok:
        print("\nEnvironment is ready.")
        return

    print("\nEnvironment not ready.")
    print("This project uses a single setup path:")
    print("  1. pyenv local 3.10.x")
    print("  2. Build .venv using that interpreter")
    print("  3. Install requirements from config.json\n")

    # Step 1: rewrite requirements from merged deps
    _, deps = _load_env_config()
    req_path = _write_requirements(deps)

    # Step 2: choose interpreter for install
    active_venv_python = _venv_python()
    inside_venv = (sys.prefix != sys.base_prefix) and active_venv_python.exists() \
                  and (Path(sys.executable).resolve() == active_venv_python.resolve())
    if inside_venv:
        pyexec = Path(sys.executable)
        LOGGER.info("Installing into active venv: %s", pyexec)
    else:
        pyexec = _setup_venv()
        if not pyexec:
            print("Could not create venv.")
            return

    # Step 3: install requirements
    extra_idx = _torch_extra_index(deps)

    print(f"Interpreter: {pyexec}")
    print("Installing dependencies...")
    if not _install_requirements(pyexec, req_path, extra_index_url=extra_idx):
        print("Installation failed.")
        return

    # Step 4: post-install verification using the same interpreter
    print("\nRechecking environment with:", pyexec)
    try:
        out = subprocess.run(
            [
                str(pyexec),
                "-c",
                "from src.cli.environment import check_environment; "
                "import sys; ok, lines = check_environment(); "
                "print('\\n'.join(lines)); sys.exit(0 if ok else 1)"
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(out.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Post-install check failed.")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return

    print("\nVenv ready.")
    print("Activate with:")
    print("  source .venv/bin/activate")
    print("Run CLI:")
    print("  python -m src.cli.main")
