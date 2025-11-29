from src.utils.logging import get_logger

logger = get_logger("S2_SUMMARY")


def log_alignment_summary(sample_size, total_images):
    """Final decision logging for S2."""
    logger.info("S2: Alignment summary:")
    logger.info("S2: Sample size used: %d", sample_size)
    logger.info("S2: Total available images: %d", total_images)
    logger.info("S2: Alignment checks: ACCEPTED")
