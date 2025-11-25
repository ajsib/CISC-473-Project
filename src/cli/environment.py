import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.logging import get_logger

LOGGER = get_logger("ENV")

# Map pip package names to import module names
MODULE_NAME_OVERRIDES: Dict[str, str] = {
    "opencv-python": "cv2",
    "PyYAML": "yaml",
    "pillow": "PIL",
    "scikit-image": "skimage",
}


def _project_root() -> Path:
    # .../project/src/cli/environment.py -> project/
    return Path(__file__).resolve().parents[2]


def _load_env_config() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load python-range and dependencies from config.json['env']."""
    root = _project_root()
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        LOGGER.error("config.json not found at '%s'.", cfg_path)
        return {}, {}

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    env_cfg = cfg.get("env", {})
    python_cfg = env_cfg.get("python", {})
    deps_cfg = env_cfg.get("dependencies", {})

    if not deps_cfg:
        LOGGER.warning("No env.dependencies section found in config.json.")
    return python_cfg, deps_cfg


def _package_to_module(pkg: str) -> str:
    if pkg in MODULE_NAME_OVERRIDES:
        return MODULE_NAME_OVERRIDES[pkg]
    return pkg.replace("-", "_")


def _parse_version(v: str | None) -> Tuple[int, int] | None:
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
    cur_major, cur_minor = sys.version_info[:2]
    cur_tuple = (cur_major, cur_minor)
    cur_str = f"{cur_major}.{cur_minor}"

    min_v = _parse_version(python_cfg.get("min"))
    max_v = _parse_version(python_cfg.get("max"))

    # Too old -> hard error
    if min_v and cur_tuple < min_v:
        return (
            False,
            f"Python {cur_str} is below configured min {min_v[0]}.{min_v[1]}",
        )

    # Above max -> warn but allow (this is your 3.14 case)
    if max_v and cur_tuple > max_v:
        return (
            True,
            f"Python {cur_str} above configured range "
            f"{min_v[0]}.{min_v[1] if min_v else '??'}–{max_v[0]}.{max_v[1]} "
            f"(untested, proceed with caution)",
        )

    if min_v and max_v:
        return (
            True,
            f"Python {cur_str} within configured range "
            f"{min_v[0]}.{min_v[1]}–{max_v[0]}.{max_v[1]}",
        )
    if min_v:
        return True, f"Python {cur_str} >= min {min_v[0]}.{min_v[1]}"
    if max_v:
        return True, f"Python {cur_str} <= max {max_v[0]}.{max_v[1]}"
    return True, f"Python {cur_str} (no explicit min/max in env.python)"


def _check_modules(deps_cfg: Dict[str, str]) -> Tuple[bool, List[str]]:
    ok = True
    lines: List[str] = []

    for pkg in deps_cfg.keys():
        mod = _package_to_module(pkg)
        try:
            importlib.import_module(mod)
            lines.append(f"OK   - import {mod}")
        except Exception as e:
            ok = False
            lines.append(f"FAIL - import {mod}: {e}")

    return ok, lines


def _check_torch_cuda() -> Tuple[bool, List[str]]:
    lines: List[str] = []
    try:
        import torch  # type: ignore

        lines.append(f"OK   - torch version: {torch.__version__}")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            lines.append(f"OK   - CUDA available, {device_count} device(s) visible")
            ok = True
        else:
            lines.append("WARN - CUDA not available; running on CPU only")
            ok = True
    except Exception as e:
        lines.append(f"FAIL - torch / CUDA check failed: {e}")
        ok = False

    return ok, lines


def _make_requirements_lines(deps_cfg: Dict[str, str]) -> List[str]:
    lines: List[str] = []
    for pkg, ver in deps_cfg.items():
        if ver:
            lines.append(f"{pkg}=={ver}")
        else:
            lines.append(pkg)
    return lines


def _write_requirements_from_config() -> Path:
    root = _project_root()
    _python_cfg, deps_cfg = _load_env_config()
    req_lines = _make_requirements_lines(deps_cfg)
    path = root / "requirements.txt"
    with path.open("w", encoding="utf-8") as f:
        for line in req_lines:
            f.write(line + "\n")
    LOGGER.info("Wrote requirements to '%s'.", path)
    return path


def _write_env_yml_from_config() -> Path:
    root = _project_root()
    python_cfg, _deps_cfg = _load_env_config()
    min_py = python_cfg.get("min", "3.10")

    req_path = root / "requirements.txt"
    if not req_path.is_file():
        _write_requirements_from_config()

    env_path = root / "env.yml"
    content = (
        "name: face-restore\n"
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        f"  - python={min_py}\n"
        "  - pip\n"
        "  - pip:\n"
        "      - -r requirements.txt\n"
    )
    with env_path.open("w", encoding="utf-8") as f:
        f.write(content)

    LOGGER.info("Wrote env.yml to '%s'.", env_path)
    return env_path


def _setup_venv_from_config() -> None:
    root = _project_root()
    python_cfg, _deps_cfg = _load_env_config()

    py_ok, py_msg = _check_python_version(python_cfg)
    (LOGGER.info if py_ok else LOGGER.error)(py_msg)
    if not py_ok:
        print(py_msg)
        print("Venv setup aborted due to Python being too old.")
        return

    venv_dir = root / ".venv"

    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        LOGGER.info("Removed existing virtual environment at '%s'.", venv_dir)

    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            cwd=root,
        )
        LOGGER.info("Created virtual environment at '%s'.", venv_dir)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to create virtual environment: %s", e)
        print(f"Failed to create virtual environment: {e}")
        return

    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    req_path = _write_requirements_from_config()

    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            cwd=root,
        )
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to upgrade pip in venv: %s", e)
        print(f"Failed to upgrade pip in venv: {e}")

    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(req_path)],
            check=True,
            cwd=root,
        )
        LOGGER.info("Installed packages in venv from '%s'.", req_path)
        print("Virtual environment setup complete.")
        if os.name == "nt":
            print("Activate with:")
            print("  .venv\\Scripts\\activate")
        else:
            print("Activate with:")
            print("  source .venv/bin/activate")
        print("Then run:")
        print("  python -m src.cli.main")
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to install packages in venv: %s", e)
        print(f"Failed to install packages in venv: {e}")


def _setup_conda_from_config() -> None:
    root = _project_root()
    env_path = _write_env_yml_from_config()

    cmd = ["conda", "env", "create", "-f", str(env_path)]
    try:
        subprocess.run(cmd, check=True, cwd=root)
        LOGGER.info("Conda environment created from '%s'.", env_path)
        print("Conda environment 'face-restore' created.")
        print("Activate and run:")
        print("  conda activate face-restore")
        print("  python -m src.cli.main")
    except FileNotFoundError:
        LOGGER.error("conda command not found on PATH.")
        print("conda is not installed or not on PATH.")
    except subprocess.CalledProcessError as e:
        LOGGER.error("conda env create failed: %s", e)
        print(f"conda env create failed: {e}")


def _clean_local_venv() -> None:
    root = _project_root()
    venv_dir = root / ".venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        LOGGER.info("Removed local virtual environment at '%s'.", venv_dir)
        print(f"Removed .venv at: {venv_dir}")
    else:
        LOGGER.info("No local .venv directory found to remove.")
        print("No .venv directory found.")


def _update_dependency_in_config() -> None:
    root = _project_root()
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        print("config.json not found, cannot update dependencies.")
        LOGGER.error("config.json not found when updating dependencies.")
        return

    pkg = input("Package name (pip name, e.g. pandas, torch, scikit-image): ").strip()
    if not pkg:
        print("Empty package name; aborting.")
        return
    ver = input("Version (empty for latest): ").strip()

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    env_cfg = cfg.setdefault("env", {})
    deps_cfg = env_cfg.setdefault("dependencies", {})

    deps_cfg[pkg] = ver
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    LOGGER.info("Updated dependency in config.json: %s==%s", pkg, ver or "<latest>")
    print(f"Updated config.env.dependencies[{pkg!r}] = {ver or '<latest>'}")


def check_environment() -> Tuple[bool, List[str]]:
    """Run environment checks based on config.env."""
    lines: List[str] = []
    python_cfg, deps_cfg = _load_env_config()

    py_ok, py_line = _check_python_version(python_cfg)
    lines.append(py_line)
    (LOGGER.info if py_ok else LOGGER.error)(py_line)

    mod_ok, mod_lines = _check_modules(deps_cfg)
    lines.extend(mod_lines)
    for line in mod_lines:
        if line.startswith("OK"):
            LOGGER.info(line)
        elif line.startswith("WARN"):
            LOGGER.warning(line)
        else:
            LOGGER.error(line)

    cuda_ok, cuda_lines = _check_torch_cuda()
    lines.extend(cuda_lines)
    for line in cuda_lines:
        if line.startswith("OK"):
            LOGGER.info(line)
        elif line.startswith("WARN"):
            LOGGER.warning(line)
        else:
            LOGGER.error(line)

    ok = py_ok and mod_ok and cuda_ok
    if ok:
        summary = "Environment check PASSED."
        LOGGER.info(summary)
        lines.append(summary)
    else:
        summary = "Environment check FAILED. See details above."
        LOGGER.error(summary)
        lines.append(summary)

    return ok, lines


def run_environment_wizard() -> None:
    """Interactive wizard to inspect and bootstrap the environment."""
    ok, lines = check_environment()

    print("\n--- Environment check ---")
    for line in lines:
        print(line)

    if ok:
        print("\nEnvironment already looks usable.")
        return

    while True:
        print("\nEnvironment is NOT ready.")
        print("Select one of the following:")
        print("[1] Setup / refresh local venv (.venv) from config.env")
        print("[2] Setup conda env from config.env via env.yml")
        print("[3] Regenerate requirements.txt and env.yml from config.env")
        print("[4] Clean local .venv directory")
        print("[5] Add or update dependency in config.env")
        print("[0] Back to main menu")
        choice = input("Choice: ").strip()

        if choice == "1":
            _setup_venv_from_config()
        elif choice == "2":
            _setup_conda_from_config()
        elif choice == "3":
            _write_requirements_from_config()
            _write_env_yml_from_config()
            print("Regenerated requirements.txt and env.yml from config.env.")
        elif choice == "4":
            _clean_local_venv()
        elif choice == "5":
            _update_dependency_in_config()
        elif choice == "0":
            break
        else:
            print("Invalid selection.")
