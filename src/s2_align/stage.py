# src/s2_align/stage.py

import os
import random
import shutil


from src.utils.logging import get_logger
from src.s1_data.utils.fs import list_image_filenames, load_metadata_frames

from src.s2_align.utils.image_checks import check_image_geometry
from src.s2_align.utils.landmark_checks import check_landmarks
from src.s2_align.utils.summary import log_alignment_summary


PRUNED_ROOT = os.path.join("results", "outputs", "s1-validated-pruned-dataset")
S2_OUT_ROOT = os.path.join("results", "outputs", "s2-processed-size-bb")


def _ensure_s2_metadata_mirror():
    """
    S2 mirrors metadata CSVs into its directory purely for convenience.
    No modification, no rewriting. Just copy from S1-pruned.
    """
    os.makedirs(S2_OUT_ROOT, exist_ok=True)

    meta_files = (
        "list_attr_celeba.csv",
        "list_eval_partition.csv",
        "list_landmarks_align_celeba.csv",
    )

    for name in meta_files:
        src = os.path.join(PRUNED_ROOT, name)
        dst = os.path.join(S2_OUT_ROOT, name)

        if not os.path.isfile(src):
            raise SystemExit(f"S2: Missing expected S1-pruned metadata file: {src}")

        if not os.path.isfile(dst):
            # copy once; do not overwrite
            shutil.copyfile(src, dst)


def run(config):
    logger = get_logger("S2")
    logger.info("S2: Alignment verification started.")

    # 1) Ensure all metadata CSVs exist under S2 output (mirrored from S1)
    _ensure_s2_metadata_mirror()

    # 2) Image directory is directly the S1-pruned images (CelebA-aligned)
    image_dir = os.path.join(PRUNED_ROOT, "img_align_celeba")
    if not os.path.isdir(image_dir):
        logger.error("S2: Expected image directory missing: %s", image_dir)
        raise SystemExit(1)

    # 3) Load all image filenames
    image_filenames = list_image_filenames(image_dir)
    if not image_filenames:
        logger.error("S2: No images found in %s", image_dir)
        raise SystemExit(1)

    # 4) Load the metadata frames from S2 mirror directory
    csv_paths = {
        "attr": os.path.join(S2_OUT_ROOT, "list_attr_celeba.csv"),
        "partition": os.path.join(S2_OUT_ROOT, "list_eval_partition.csv"),
        "landmarks": os.path.join(S2_OUT_ROOT, "list_landmarks_align_celeba.csv"),
    }

    frames = load_metadata_frames(csv_paths)
    landmarks_df = frames["landmarks"]

    # 5) Sample images for QC
    total = len(image_filenames)
    sample_size = min(250, total)
    sample_list = random.sample(sorted(list(image_filenames)), sample_size)

    logger.info(
        "S2: Using %d sampled images (out of %d total) for verification.",
        sample_size, total
    )

    # 6) CelebA-aligned images: expected fixed raw dimensions from config
    align_cfg = config["data"]["alignment"]["params"]
    expected_w = align_cfg["expected_width"]
    expected_h = align_cfg["expected_height"]

    # 7) Image geometry check
    ok_geometry = check_image_geometry(
        image_dir=image_dir,
        samples=sample_list,
        expected_w=expected_w,
        expected_h=expected_h,
    )
    if not ok_geometry:
        logger.error("S2: Image geometry verification FAILED.")
        raise SystemExit(1)

    # 8) Landmark check
    ok_landmarks = check_landmarks(
        image_dir=image_dir,
        samples=sample_list,
        df_landmarks=landmarks_df,
        expected_w=expected_w,
        expected_h=expected_h,
    )
    if not ok_landmarks:
        logger.error("S2: Landmark verification FAILED.")
        raise SystemExit(1)

    # 9) Summary and exit
    log_alignment_summary(sample_size=sample_size, total_images=total)
    logger.info("S2: Alignment verification completed successfully.")
