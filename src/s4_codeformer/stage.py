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
