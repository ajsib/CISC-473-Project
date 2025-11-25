# src/s1_data/stage.py

import os

from src.utils.logging import get_logger
from src.s1_data.utils.fs import (
    ensure_required_paths,
    list_image_filenames,
    load_metadata_frames,
)
from src.s1_data.utils.validators import (
    validate_csv_schemas,
    validate_id_consistency,
    validate_partitions,
    log_dataset_summary,
)


def run(config):
    logger = get_logger("S1")
    logger.info("S1: Data ingestion and verification started.")

    # 1. Resolve expected paths
    data_root = "data"
    image_dir = os.path.join(data_root, "img_align_celeba")

    csv_paths = {
        "attr": os.path.join(data_root, "list_attr_celeba.csv"),
        "bbox": os.path.join(data_root, "list_bbox_celeba.csv"),
        "partition": os.path.join(data_root, "list_eval_partition.csv"),
        "landmarks": os.path.join(data_root, "list_landmarks_align_celeba.csv"),
    }

    # 2. Check that all required directories and files exist
    ensure_required_paths(image_dir, csv_paths)

    # 3. Scan image filenames
    image_filenames = list_image_filenames(image_dir)
    if not image_filenames:
        logger.error("S1: No image files found under '%s'.", image_dir)
        raise SystemExit(1)

    logger.info(
        "S1: Found %d image files under '%s'.",
        len(image_filenames),
        image_dir,
    )

    # 4. Load metadata CSVs into pandas DataFrames
    metadata_frames = load_metadata_frames(csv_paths)

    # 5. Validate CSV schemas (very lightweight structural checks)
    validate_csv_schemas(metadata_frames)

    # 6. Validate ID consistency between images and each CSV
    validate_id_consistency(image_filenames, metadata_frames)

    # 7. Validate partition labels and basic split structure
    validate_partitions(metadata_frames["partition"])

    # 8. Log a concise dataset summary for downstream reference
    log_dataset_summary(image_filenames, metadata_frames)

    logger.info("S1: Data ingestion and verification completed successfully.")
