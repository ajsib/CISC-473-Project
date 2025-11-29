import os
import shutil

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
from src.s1_data.utils.prune import prune_dataset

def run(config):
    logger = get_logger("S1")
    logger.info("S1: Data ingestion and verification started.")

    # 1. Resolve expected paths
    data_root = "data"
    target_root = "results/outputs"
    image_dir = os.path.join(data_root, "img_align_celeba")
    pruned_root = os.path.join(target_root, "s1-validated-pruned-dataset")
    pruned_img_dir = os.path.join(pruned_root, "img_align_celeba")
    os.makedirs(pruned_img_dir, exist_ok=True)

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

    # 7.5. Prompt for pruning size
    min_size = 100
    max_size = len(image_filenames)
    prune_input = input(f"S1: Enter sample/prune size (between {min_size} and {max_size}, or Enter for full set): ").strip()
    sample_size = None
    prune_seed = 1337

    if prune_input:
        sample_size = int(prune_input)
        if not (min_size <= sample_size <= max_size):
            logger.error(f"S1: Sample size {sample_size} outside range [{min_size},{max_size}].")
            raise SystemExit(1)
        logger.info("S1: Pruning dataset to %d samples...", sample_size)
        prune_dataset(
            image_filenames=image_filenames,
            metadata_frames=metadata_frames,
            csv_paths=csv_paths,
            image_dir=image_dir,
            out_dir=pruned_root,
            sample_size=sample_size,
            seed=prune_seed,
        )
        logger.info("S1: Pruned dataset written to '%s'.", pruned_root)
    else:
        # Copy everything (full set) to pruned-dataset for uniformity
        logger.info("S1: Copying full dataset to pruned-dataset (no sampling)...")
        # Copy images
        for img_id in image_filenames:
            src = os.path.join(image_dir, img_id)
            dst = os.path.join(pruned_img_dir, img_id)
            shutil.copyfile(src, dst)
        # Copy CSVs
        for key, src_csv in csv_paths.items():
            dst_csv = os.path.join(pruned_root, os.path.basename(src_csv))
            shutil.copyfile(src_csv, dst_csv)
        logger.info("S1: Full dataset copied to '%s'.", pruned_root)

    # 8. Log a concise dataset summary for downstream reference
    log_dataset_summary(image_filenames, metadata_frames)
    logger.info("S1: Data ingestion and verification completed successfully.")
