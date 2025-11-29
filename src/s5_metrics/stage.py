import os

from src.utils.logging import get_logger


def run(config):
    logger = get_logger("S5")
    logger.info("S5: Metrics computation (placeholder).")

    tables_root = os.path.join("results", "tables")
    os.makedirs(tables_root, exist_ok=True)
    logger.info("S5: Ensured tables directory exists at '%s'.", tables_root)
    logger.info("S5: No metrics computed yet (stub implementation).")
