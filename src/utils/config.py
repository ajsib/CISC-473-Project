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
