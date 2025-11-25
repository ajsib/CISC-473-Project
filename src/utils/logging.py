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
