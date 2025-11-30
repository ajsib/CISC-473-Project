# src/s2_align/stage.py

import os
import random
import shutil

from src.utils.logging import get_logger

from src.s1_data.utils.fs import (
    list_image_filenames,
    load_metadata_frames,
)

from src.s2_align.utils.image_checks import check_image_geometry
from src.s2_align.utils.landmark_checks import check_landmarks
from src.s2_align.utils.bbox_checks import check_bboxes
from src.s2_align.utils.summary import log_alignment_summary

# One-time alignment preprocessor (raw -> aligned 256×256)
from src.s2_align.s2a import run_s2a

# One-time bbox-space transformer (raw -> aligned)
from src.s2_align.s2b import run_s2b


PRUNED_ROOT = os.path.join("results", "outputs", "s1-validated-pruned-dataset")
S2_OUT_ROOT = os.path.join("results", "outputs", "s2-processed-size-bb")


def _ensure_s2_metadata_mirror():
    """
    Copy metadata CSVs once from S1-pruned into S2 output directory.
    No modifications, no rewriting.
    """
    os.makedirs(S2_OUT_ROOT, exist_ok=True)

    meta_files = (
        "list_attr_celeba.csv",
        "list_eval_partition.csv",
        "list_landmarks_align_celeba.csv",
        "list_bbox_celeba.csv",
    )

    for name in meta_files:
        src = os.path.join(PRUNED_ROOT, name)
        dst = os.path.join(S2_OUT_ROOT, name)

        if not os.path.isfile(src):
            raise SystemExit(f"S2: Missing expected S1-pruned file: {src}")

        if not os.path.isfile(dst):
            shutil.copyfile(src, dst)


def _detect_s2_images():
    """
    Detect whether 256×256 aligned images already exist in S2 output.
    If not, use S1-pruned raw CelebA images.
    """
    aligned_dir = os.path.join(S2_OUT_ROOT, "img_align_celeba")
    raw_dir = os.path.join(PRUNED_ROOT, "img_align_celeba")

    if os.path.isdir(aligned_dir):
        return aligned_dir, True  # already aligned

    if os.path.isdir(raw_dir):
        return raw_dir, False  # raw, requires S2A

    return None, False


def run(config):
    logger = get_logger("S2")
    logger.info("S2: Stage started.")

    # ------------------------------------------------------------
    # Ensure metadata exists in S2 output
    # ------------------------------------------------------------
    _ensure_s2_metadata_mirror()

    # ------------------------------------------------------------
    # Detect source directory
    # ------------------------------------------------------------
    image_dir, already_aligned = _detect_s2_images()
    if image_dir is None:
        logger.error("S2: No image directory available.")
        raise SystemExit(1)

    # ------------------------------------------------------------
    # Run S2A if needed
    # ------------------------------------------------------------
    if not already_aligned:
        logger.warning(
            "S2: Raw CelebA images detected at '%s'. Running S2A.",
            image_dir,
        )
        run_s2a()
        image_dir = os.path.join(S2_OUT_ROOT, "img_align_celeba")
        logger.info("S2: Using aligned directory: %s", image_dir)

    # ------------------------------------------------------------
    # Metadata paths (S2 output)
    # ------------------------------------------------------------
    csv_paths = {
        "attr": os.path.join(S2_OUT_ROOT, "list_attr_celeba.csv"),
        "bbox": os.path.join(S2_OUT_ROOT, "list_bbox_celeba.csv"),
        "partition": os.path.join(S2_OUT_ROOT, "list_eval_partition.csv"),
        "landmarks": os.path.join(S2_OUT_ROOT, "list_landmarks_align_celeba.csv"),
    }

    # ------------------------------------------------------------
    # Load filenames
    # ------------------------------------------------------------
    image_filenames = list_image_filenames(image_dir)
    if not image_filenames:
        logger.error("S2: No images under '%s'.", image_dir)
        raise SystemExit(1)

    # ------------------------------------------------------------
    # Load metadata frames
    # ------------------------------------------------------------
    frames = load_metadata_frames(csv_paths)
    landmarks_df = frames["landmarks"]
    bbox_df = frames["bbox"]

    # ------------------------------------------------------------
    # Sample images for QC
    # ------------------------------------------------------------
    total = len(image_filenames)
    sample_size = min(250, total)
    sample_list = random.sample(sorted(list(image_filenames)), sample_size)

    logger.info(
        "S2: Sampling %d images (out of %d) for QC.",
        sample_size,
        total,
    )

    # ------------------------------------------------------------
    # Expected aligned resolution from config
    # ------------------------------------------------------------
    alignment_cfg = config["data"]["alignment"]["params"]
    target_size = alignment_cfg["image_size"]

    expected_w = target_size
    expected_h = target_size

    # ------------------------------------------------------------
    # GEOMETRY CHECK
    # ------------------------------------------------------------
    ok_geometry = check_image_geometry(
        image_dir=image_dir,
        samples=sample_list,
        expected_w=expected_w,
        expected_h=expected_h,
    )
    if not ok_geometry:
        logger.error("S2: Geometry verification failed.")
        raise SystemExit(1)

    # ------------------------------------------------------------
    # LANDMARK CHECK
    # ------------------------------------------------------------
    ok_landmarks = check_landmarks(
        image_dir=image_dir,
        samples=sample_list,
        df_landmarks=landmarks_df,
        expected_w=expected_w,
        expected_h=expected_h,
    )
    if not ok_landmarks:
        logger.error("S2: Landmark verification failed.")
        raise SystemExit(1)

    # ------------------------------------------------------------
    # BBOX CHECK
    # ------------------------------------------------------------
    ok_bboxes = check_bboxes(
        image_dir=image_dir,
        samples=sample_list,
        df_bbox=bbox_df,
        expected_w=expected_w,
        expected_h=expected_h,
    )

    if not ok_bboxes:
        logger.warning(
            "S2: Bounding boxes appear to be raw-space. Running S2B."
        )

        run_s2b()  # produce aligned-space bbox CSV

        # Reload updated bbox CSV
        frames = load_metadata_frames(csv_paths)
        bbox_df = frames["bbox"]

        ok_bboxes = check_bboxes(
            image_dir=image_dir,
            samples=sample_list,
            df_bbox=bbox_df,
            expected_w=expected_w,
            expected_h=expected_h,
        )

        if not ok_bboxes:
            logger.error("S2: Bbox check failed after S2B.")
            raise SystemExit(1)

        logger.info("S2: Bbox check passed after S2B transform.")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    log_alignment_summary(sample_size=sample_size, total_images=total)
    logger.info("S2: Completed successfully.")
