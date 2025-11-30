import csv
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.s3_degrade.utils.io import (
    ensure_dir,
    list_aligned_filenames,
    load_image,
    save_image_atomic,
    verify_rgb_images_ok,
    list_valid_rgb_images,
)
from src.s3_degrade.utils.degradations import apply_degradation

MANIFEST_FILENAME = "s3_degrade_manifest.csv"
PATCH_ID = "S3 harden v2025-11-27-2005"


def _load_partition_map(partition_csv: str) -> Dict[str, int]:
    if not os.path.isfile(partition_csv):
        raise FileNotFoundError(f"S3: partition CSV not found: {partition_csv}")
    df = pd.read_csv(partition_csv, sep=r"\s+|,", engine="python", comment="#")
    id_col, split_col = df.columns[0], df.columns[1]
    mapping: Dict[str, int] = {}
    for _, row in df.iterrows():
        mapping[str(row[id_col])] = int(row[split_col])
    return mapping


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run(config):
    logger = get_logger("S3")
    logger.info("S3: %s", PATCH_ID)
    logger.info("S3: Synthetic degradation stage started.")

    # Inputs come from S1-pruned CelebA-aligned images
    s2_root = os.path.join("results", "outputs", "s2-processed-size-bb")
    aligned_root = os.path.join(s2_root, "img_align_celeba")
    partition_csv = os.path.join(s2_root, "list_eval_partition.csv")

    if not os.path.isdir(aligned_root):
        logger.error("S3: Expected aligned root '%s' not found.", aligned_root)
        raise SystemExit(1)
    if not os.path.isfile(partition_csv):
        logger.error("S3: Expected partition CSV '%s' not found.", partition_csv)
        raise SystemExit(1)

    try:
        degr_cfg = config["data"]["degradations"]
        presets: List[Dict] = degr_cfg["presets"]
        seed = int(degr_cfg["seed"])
        raw_output_size = degr_cfg.get("output_size", None)
        output_size = int(raw_output_size) if raw_output_size is not None else None
    except KeyError as e:
        logger.error("S3: Missing configuration key in config['data']['degradations']: %s", e)
        raise SystemExit(1)
    if not presets:
        logger.error("S3: No degradation presets defined in config.json.")
        raise SystemExit(1)

    logger.info("S3: Using %d degradation presets.", len(presets))
    if output_size is None:
        logger.info("S3: Global seed=%d, output_size=None (keep source geometry)", seed)
    else:
        logger.info("S3: Global seed=%d, output_size=%d", seed, output_size)
    _seed_all(seed)

    # Outputs go under results/outputs/s3-degrade/<preset>/imgs
    lq_root = os.path.join("results", "outputs", "s3-degrade")
    logs_root = os.path.join("results", "logs")
    ensure_dir(lq_root)
    ensure_dir(logs_root)
    manifest_path = os.path.join(logs_root, MANIFEST_FILENAME)
    expect_size = (output_size, output_size) if output_size is not None else None

    filenames = list_aligned_filenames(aligned_root)
    if not filenames:
        logger.error("S3: No aligned images found in '%s'.", aligned_root)
        raise SystemExit(1)
    total_images = len(filenames)
    logger.info("S3: Found %d aligned images.", total_images)

    partition_map = _load_partition_map(partition_csv)
    logger.info("S3: Loaded partition map for %d entries from '%s'.", len(partition_map), partition_csv)

    try:
        schema = config["data"]["manifests"]["schema_csv"]
    except KeyError:
        schema = ["id", "path_gt", "path_deg", "degradation", "split"]

    required = {"id", "path_gt", "path_deg", "degradation", "split"}
    if set(schema) != required:
        logger.warning("S3: Manifest schema_csv differs from default. Expected %s, got %s.",
                       sorted(list(required)), schema)

    all_rows: List[Dict] = []

    for preset_idx, preset in enumerate(presets):
        preset_name = preset.get("name", f"preset_{preset_idx}")
        out_dir = os.path.join(lq_root, preset_name, "imgs")
        ensure_dir(out_dir)

        valid_now = set(list_valid_rgb_images(out_dir, expect_size=expect_size))
        existing_files = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
        missing = [fn for fn in filenames if fn not in existing_files]
        corrupt = [fn for fn in existing_files if fn not in valid_now]

        logger.info(
            "S3: Preset '%s': existing=%d, valid=%d, missing=%d, corrupt=%d (expected=%d).",
            preset_name, len(existing_files), len(valid_now), len(missing), len(corrupt), total_images
        )

        to_build = set(missing) | set(corrupt)
        if not to_build:
            logger.info("S3: Preset '%s' already complete and valid. Skipping rebuild.", preset_name)
        else:
            logger.info("S3: Starting preset '%s' build for %d files -> %s", preset_name, len(to_build), out_dir)

            processed_this_run = 0
            missing_partition = 0

            for i, fname in enumerate(filenames, 1):
                if fname not in to_build:
                    continue

                image_id = fname
                path_gt = os.path.join(aligned_root, fname)
                path_deg = os.path.join(out_dir, fname)

                split = partition_map.get(image_id)
                if split is None:
                    missing_partition += 1
                    if missing_partition <= 5:
                        logger.warning("S3: No partition entry for '%s' in '%s'. Skipping.", image_id, partition_csv)
                    continue

                try:
                    img_gt = load_image(path_gt)  # PIL RGB
                except Exception as e:
                    logger.error("S3: Failed to load GT image '%s': %s", path_gt, e)
                    continue

                try:
                    img_lq = apply_degradation(img_gt, preset, output_size=output_size)
                except Exception as e:
                    logger.error("S3: Degradation failed for '%s' under preset '%s': %s", image_id, preset_name, e)
                    continue

                try:
                    save_image_atomic(img_lq, path_deg)
                except Exception as e:
                    logger.error("S3: Failed to save LQ image '%s': %s", path_deg, e)
                    continue

                processed_this_run += 1
                if processed_this_run % 1000 == 0:
                    logger.info("S3: Preset '%s': built %d files...", preset_name, processed_this_run)

            logger.info(
                "S3: Finished preset '%s' build. Added/rewritten this run: %d. Missing partition entries: %d.",
                preset_name, processed_this_run, missing_partition
            )

        valid_files = list(list_valid_rgb_images(out_dir, expect_size=expect_size))
        for fname in valid_files:
            split = partition_map.get(fname)
            if split is None:
                continue
            all_rows.append({
                "id": fname,
                "path_gt": os.path.join(aligned_root, fname),
                "path_deg": os.path.join(out_dir, fname),
                "degradation": preset_name,
                "split": int(split),
            })
        logger.info("S3: Preset '%s': %d valid files recorded to manifest.", preset_name, len(valid_files))

    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    os.replace(tmp_manifest, manifest_path)

    deg_paths = [row["path_deg"] for row in all_rows]
    ok_count = verify_rgb_images_ok(deg_paths, expect_size=expect_size)
    if ok_count != len(deg_paths):
        logger.error(
            "S3: Sanity check failed. Manifest rows=%d, valid LQ files=%d.",
            len(deg_paths), ok_count
        )
        raise SystemExit(1)

    logger.info("S3: Wrote canonical manifest: %s (rows=%d)", manifest_path, len(all_rows))
    logger.info("S3: Sanity check passed. All %d LQ files referenced in manifest are valid.", ok_count)
    logger.info("S3: Synthetic degradation stage completed successfully.")
