# Run this from the project root

# 1. Create directories
mkdir -p src/cli \
         src/s1_data \
         src/s2_align \
         src/s3_degrade \
         src/s4_gfpgan \
         src/s4_codeformer \
         src/s5_metrics \
         src/s6_figures \
         src/s7_logging \
         src/utils \
         results/logs

# 2. Remove legacy single-file stage scripts (now replaced by packages)
rm -f src/degrade.py src/detect_align.py src/metrics_identity.py \
      src/metrics_lpips.py src/metrics_pixel.py src/run_codeformer.py \
      src/run_gfpgan.py

# 3. Ensure src is a package
cat > src/__init__.py << 'EOF'
"""Top-level package for the face restoration pipeline."""
EOF

# 4. CLI package __init__
cat > src/cli/__init__.py << 'EOF'
"""CLI package for the face restoration pipeline."""
EOF

# 5. CLI main entry point
cat > src/cli/main.py << 'EOF'
import argparse
import sys

from src.utils.config import load_config
from src.utils.logging import get_logger

from src.s1_data.stage import run as run_s1
from src.s2_align.stage import run as run_s2
from src.s3_degrade.stage import run as run_s3
from src.s4_gfpgan.stage import run as run_s4_gfpgan
from src.s4_codeformer.stage import run as run_s4_codeformer
from src.s5_metrics.stage import run as run_s5
from src.s6_figures.stage import run as run_s6
from src.s7_logging.stage import run as run_s7


STAGE_FUNCS = {
    "s1": run_s1,
    "s2": run_s2,
    "s3": run_s3,
    "s4_gfpgan": run_s4_gfpgan,
    "s4_codeformer": run_s4_codeformer,
    "s5": run_s5,
    "s6": run_s6,
    "s7": run_s7,
}

STAGE_ORDER = ["s1", "s2", "s3", "s4_gfpgan", "s4_codeformer", "s5", "s6", "s7"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Face restoration pipeline CLI (GFPGAN vs CodeFormer)."
    )
    parser.add_argument(
        "--stage",
        choices=["all"] + STAGE_ORDER,
        default="all",
        help=(
            "Stage to run. "
            "Use 'all' (default) to run the full S1–S7 pipeline, "
            "or choose a single stage like 's3' or 's4_gfpgan'."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = get_logger("CLI")

    logger.info("Loading configuration from config.json")
    config = load_config("config.json")

    if args.stage == "all":
        stages_to_run = STAGE_ORDER
        logger.info("Running full pipeline: %s", ", ".join(STAGE_ORDER))
    else:
        stages_to_run = [args.stage]
        logger.info("Running single stage: %s", args.stage)

    for stage_name in stages_to_run:
        stage_func = STAGE_FUNCS[stage_name]
        logger.info("=== START %s ===", stage_name.upper())
        stage_func(config)
        logger.info("=== END   %s ===", stage_name.upper())

    logger.info("Pipeline execution completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF

# 6. utils package __init__
cat > src/utils/__init__.py << 'EOF'
"""Shared utilities for configuration, logging, and I/O."""
EOF

# 7. utils/config.py
cat > src/utils/config.py << 'EOF'
import json
import os
from typing import Any, Dict


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load the global configuration from JSON.

    Exits the program with a clear message if the file is missing or invalid.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"[CONFIG] config file not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[CONFIG] Failed to parse JSON at {path}: {e}") from e

    return cfg
EOF

# 8. utils/logging.py
cat > src/utils/logging.py << 'EOF'
import logging
import os
from typing import Dict

_LOGGERS: Dict[str, logging.Logger] = {}


def _ensure_log_dir() -> str:
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_logger(name: str) -> logging.Logger:
    """Return a logger with console + file handlers attached.

    All logs go to results/logs/pipeline.log plus stdout.
    Handlers are attached only once per logger name.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    _ensure_log_dir()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_path = os.path.join("results", "logs", "pipeline.log")
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
EOF

# 9. S1 Data package + stage
cat > src/s1_data/__init__.py << 'EOF'
"""S1: Data ingestion and verification."""
EOF

cat > src/s1_data/stage.py << 'EOF'
import os

from src.utils.logging import get_logger


def run(config):
    logger = get_logger("S1")
    logger.info("S1: Data ingestion and verification (placeholder).")

    data_dir = os.path.join("data", "img_align_celeba")
    if not os.path.isdir(data_dir):
        logger.error("S1: Expected data directory '%s' not found.", data_dir)
        raise SystemExit(1)

    logger.info("S1: Found data directory '%s'.", data_dir)
    logger.info("S1: Placeholder stage completed successfully.")
EOF

# 10. S2 Align package + stage
cat > src/s2_align/__init__.py << 'EOF'
"""S2: Alignment verification (no-op placeholder)."""
EOF

cat > src/s2_align/stage.py << 'EOF'
from src.utils.logging import get_logger


def run(config):
    logger = get_logger("S2")
    logger.info("S2: Alignment verification (placeholder, no-op).")
    logger.info("S2: Assuming CelebA is already aligned to 256x256.")
EOF

# 11. S3 Degrade package + stage
cat > src/s3_degrade/__init__.py << 'EOF'
"""S3: Synthetic degradation (placeholder)."""
EOF

cat > src/s3_degrade/stage.py << 'EOF'
import os

from src.utils.logging import get_logger


def run(config):
    logger = get_logger("S3")
    logger.info("S3: Synthetic degradation (placeholder).")

    outputs_root = os.path.join("results", "outputs", "lq")
    os.makedirs(outputs_root, exist_ok=True)
    logger.info("S3: Ensured output directory exists at '%s'.", outputs_root)
    logger.info("S3: No degradations applied yet (stub implementation).")
EOF

# 12. S4 GFPGAN package + stage
cat > src/s4_gfpgan/__init__.py << 'EOF'
"""S4A: GFPGAN inference (placeholder)."""
EOF

cat > src/s4_gfpgan/stage.py << 'EOF'
import os

from src.utils.logging import get_logger


def run(config):
    logger = get_logger("S4_GFP")
    logger.info("S4A: GFPGAN inference (placeholder).")

    outputs_root = os.path.join("results", "outputs", "gfpgan")
    os.makedirs(outputs_root, exist_ok=True)
    logger.info("S4A: Ensured output directory exists at '%s'.", outputs_root)
    logger.info("S4A: No model inference run yet (stub implementation).")
EOF

# 13. S4 CodeFormer package + stage
cat > src/s4_codeformer/__init__.py << 'EOF'
"""S4B: CodeFormer inference (placeholder)."""
EOF

cat > src/s4_codeformer/stage.py << 'EOF'
import os

from src.utils.logging import get_logger


def run(config):
    logger = get_logger("S4_CF")
    logger.info("S4B: CodeFormer inference (placeholder).")

    outputs_root = os.path.join("results", "outputs", "codeformer")
    os.makedirs(outputs_root, exist_ok=True)
    logger.info("S4B: Ensured output directory exists at '%s'.", outputs_root)

    fidelity = config.get("upstreams", {}).get("codeformer", {}).get(
        "fidelity_knob", {}
    ).get("default", 0.5)
    logger.info("S4B: Stub using default fidelity knob w=%s.", fidelity)
    logger.info("S4B: No model inference run yet (stub implementation).")
EOF

# 14. S5 Metrics package + stage
cat > src/s5_metrics/__init__.py << 'EOF'
"""S5: Metrics computation (placeholder)."""
EOF

cat > src/s5_metrics/stage.py << 'EOF'
import os

from src.utils.logging import get_logger


def run(config):
    logger = get_logger("S5")
    logger.info("S5: Metrics computation (placeholder).")

    tables_root = os.path.join("results", "tables")
    os.makedirs(tables_root, exist_ok=True)
    logger.info("S5: Ensured tables directory exists at '%s'.", tables_root)
    logger.info("S5: No metrics computed yet (stub implementation).")
EOF

# 15. S6 Figures package + stage
cat > src/s6_figures/__init__.py << 'EOF'
"""S6: Figure generation (placeholder)."""
EOF

cat > src/s6_figures/stage.py << 'EOF'
import os

from src.utils.logging import get_logger


def run(config):
    logger = get_logger("S6")
    logger.info("S6: Figure generation (placeholder).")

    figures_root = os.path.join("results", "figures")
    os.makedirs(figures_root, exist_ok=True)
    logger.info("S6: Ensured figures directory exists at '%s'.", figures_root)
    logger.info("S6: No figures generated yet (stub implementation).")
EOF

# 16. S7 Logging package + stage
cat > src/s7_logging/__init__.py << 'EOF'
"""S7: Run manifest and provenance (placeholder)."""
EOF

cat > src/s7_logging/stage.py << 'EOF'
import json
import os
from datetime import datetime

from src.utils.logging import get_logger


def run(config):
    logger = get_logger("S7")
    logger.info("S7: Run manifest and provenance (placeholder).")

    logs_root = os.path.join("results", "logs")
    os.makedirs(logs_root, exist_ok=True)
    manifest_path = os.path.join(logs_root, "run_manifest.json")

    manifest = {
        "project_name": config.get("project_name", "unknown-project"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "note": "Placeholder manifest written by S7 stub.",
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("S7: Wrote placeholder manifest to '%s'.", manifest_path)
EOF

# 17. Rewrite env.yml (same content, just ensured)
cat > env.yml << 'EOF'
name: face-restore
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.3.1
  - torchvision=0.18.1
  - pytorch-cuda=11.8  # change to 12.1 if needed
  - pip
  - pip:
      - facenet-pytorch==2.6.0
      - lpips==0.1.4
      - opencv-python==4.10.0.84
      - scikit-image==0.24.0
      - pandas==2.2.2
      - PyYAML==6.0.2
      - tqdm==4.66.4
      - matplotlib==3.9.0
      - pillow==10.4.0
EOF

# 18. Rewrite config.json (same as current, just ensured)
cat > config.json << 'EOF'
{
  "project_name": "Face Restoration: GFPGAN vs CodeFormer",
  "visibility": "public",
  "license": "MIT",
  "contributors": [
    {"name": "Monica Stef", "role": "co-author"},
    {"name": "Enqi Liang", "role": "co-author"},
    {"name": "Aidan Sibley", "role": "co-author"}
  ],
  "stack": {
    "language": "Python>=3.10,<3.12",
    "framework": "PyTorch",
    "versions": {
      "torch": "2.3.1",
      "torchvision": "0.18.1",
      "cuda": ["11.8", "12.1"],
      "python_packages": {
        "facenet-pytorch": "2.6.0",
        "lpips": "0.1.4",
        "opencv-python": "4.10.0.84",
        "scikit-image": "0.24.0",
        "pandas": "2.2.2",
        "PyYAML": "6.0.2",
        "tqdm": "4.66.4",
        "matplotlib": "3.9.0",
        "pillow": "10.4.0"
      }
    },
    "env_management": "conda",
    "env_file": "env.yml",
    "determinism": {
      "seed": 1337,
      "torch.backends.cudnn.benchmark": false,
      "torch.backends.cudnn.deterministic": true,
      "numpy_seed": 1337,
      "python_hash_seed": "0"
    },
    "execution_targets": ["Local GPU", "Google Colab", "Kaggle", "CPU Fallback"]
  },
  "upstreams": {
    "gfpgan": {
      "repo": "https://github.com/TencentARC/GFPGAN",
      "default_checkpoint": "GFPGANv1.4",
      "commit_pin": "to-fill"
    },
    "codeformer": {
      "repo": "https://github.com/sczhou/CodeFormer",
      "default_checkpoint": "codeformer-v0.1.0",
      "commit_pin": "to-fill",
      "fidelity_knob": {"name": "w", "range": [0.3, 0.9], "default": 0.5}
    }
  },
  "data": {
    "roots": {
      "raw": "data/raw",
      "processed": "data/processed",
      "historical": "data/historical",
      "manifests": "data/manifests"
    },
    "alignment": {
      "detector": "MTCNN",
      "params": {"image_size": 256, "margin": 20, "postprocess": "center-crop-then-resize"},
      "policy": "same pipeline for input and restored outputs before metrics"
    },
    "degradations": {
      "presets": [
        {"name": "gauss_blur_sigma_0.5", "type": "gaussian_blur", "sigma": 0.5},
        {"name": "gauss_blur_sigma_1.0", "type": "gaussian_blur", "sigma": 1.0},
        {"name": "jpeg_q10", "type": "jpeg", "quality": 10},
        {"name": "jpeg_q30", "type": "jpeg", "quality": 30},
        {"name": "noise_sigma_5", "type": "gaussian_noise", "sigma": 5.0},
        {"name": "noise_sigma_15", "type": "gaussian_noise", "sigma": 15.0}
      ],
      "seed": 1337,
      "output_size": 256
    },
    "manifests": {
      "schema_csv": ["id", "path_gt", "path_deg", "degradation", "split"]
    },
    "governance": {
      "license_requirement": "public-domain or permissive",
      "pii_policy": "no non-consensual personal photos; anonymize filenames; keep urls.txt"
    }
  },
  "evaluation": {
    "paired_metrics": ["PSNR", "SSIM", "LPIPS"],
    "unpaired_metrics": ["ArcFaceCosine"],
    "tables": {"dir": "results/tables", "schema_csv": ["method", "preset", "metric", "mean", "std", "n"]},
    "figures": {"dir": "results/figures", "order": ["input", "gfpgan", "codeformer"]}
  },
  "experiments": {
    "matrix": {
      "methods": ["gfpgan", "codeformer"],
      "degradations": ["gauss_blur_sigma_0.5", "gauss_blur_sigma_1.0", "jpeg_q10", "jpeg_q30", "noise_sigma_5", "noise_sigma_15"],
      "codeformer_fidelity_w": [0.3, 0.5, 0.7, 0.9]
    }
  }
}
EOF
