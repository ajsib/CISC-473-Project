# src/s4_gfpgan/stage.py

import os
from collections import defaultdict

import pandas as pd

from src.utils.logging import get_logger
from src.s4_gfpgan.utils.io import (
    ensure_dir,
    read_manifest_csv,
    load_image_rgb,
    save_image_jpeg,
    count_existing,
)
from src.s4_gfpgan.utils.model import load_gfpgan, enhance_aligned_pil


def run(config):
    logger = get_logger("S4A")
    logger.info("S4A: GFPGAN inference stage started.")

    # ------------------------------------------------------------------------------
    # Resolve S3 manifest
    # ------------------------------------------------------------------------------
    s3_manifest = os.path.join("results", "logs", "s3_degrade_manifest.csv")
    df = read_manifest_csv(s3_manifest)

    # Optional filter by split if desired later; for now process all rows.
    total_rows = len(df)
    if total_rows == 0:
        logger.error("S4A: S3 manifest is empty: %s", s3_manifest)
        logger.error("S4A: Hint: restart the CLI or ensure S3 stage was reloaded to regenerate the manifest.")
        raise SystemExit(1)

    # ------------------------------------------------------------------------------
    # Prepare output roots and load model
    # ------------------------------------------------------------------------------
    outputs_root = os.path.join("results", "outputs", "s4-gfpgan")
    logs_root = os.path.join("results", "logs")
    ensure_dir(outputs_root)
    ensure_dir(logs_root)

    s4_manifest_path = os.path.join(logs_root, "s4_gfpgan_manifest.csv")

    # Load GFPGAN model
    gcfg = config.get("upstreams", {}).get("gfpgan", {})
    ckpt_name = gcfg.get("default_checkpoint", "GFPGANv1.4")

    restorer = load_gfpgan(ckpt_name=ckpt_name, upscale=1)
    if restorer is None:
        # Dependencies missing; fail with actionable message
        raise SystemExit(1)

    # ------------------------------------------------------------------------------
    # Group by degradation preset for directory layout
    # ------------------------------------------------------------------------------
    by_preset = defaultdict(list)
    for row in df.itertuples(index=False):
        # row has attributes matching column names
        by_preset[getattr(row, "degradation")].append(row)

    logger.info(
        "S4A: Inference will run over %d presets. Total samples: %d.",
        len(by_preset),
        total_rows,
    )

    # ------------------------------------------------------------------------------
    # Per-preset inference
    # ------------------------------------------------------------------------------
    manifest_rows = []
    processed_total = 0

    for preset, rows in by_preset.items():
        out_dir = os.path.join(outputs_root, preset, "imgs")
        ensure_dir(out_dir)

        logger.info(
            "S4A: Processing preset '%s' with %d images -> %s",
            preset,
            len(rows),
            out_dir,
        )

        for i, row in enumerate(rows, 1):
            image_id = getattr(row, "id")
            path_gt = getattr(row, "path_gt")
            path_deg = getattr(row, "path_deg")
            split = int(getattr(row, "split"))

            # Load degraded input
            try:
                img_in = load_image_rgb(path_deg)
            except Exception as e:
                logger.error("S4A: Failed to load degraded '%s': %s", path_deg, e)
                continue

            # Inference
            try:
                img_restored = enhance_aligned_pil(restorer, img_in, enforce_size=None)
            except Exception as e:
                logger.error("S4A: GFPGAN failed on '%s': %s", path_deg, e)
                continue

            # Save
            path_restored = os.path.join(out_dir, image_id)
            try:
                save_image_jpeg(img_restored, path_restored, quality=95)
            except Exception as e:
                logger.error("S4A: Save failed '%s': %s", path_restored, e)
                continue

            # Manifest row
            manifest_rows.append(
                {
                    "method": "gfpgan",
                    "id": image_id,
                    "path_gt": path_gt,
                    "path_deg": path_deg,
                    "path_restored": path_restored,
                    "degradation": preset,
                    "split": split,
                    "restored_w": img_restored.width,
                    "restored_h": img_restored.height,
                }
            )

            processed_total += 1
            if processed_total % 1000 == 0:
                logger.info("S4A: Processed %d / %d total samples...", processed_total, total_rows)

        logger.info(
            "S4A: Finished preset '%s'. Processed in this group: %d.",
            preset,
            len(rows),
        )

    # ------------------------------------------------------------------------------
    # Write S4A manifest
    # ------------------------------------------------------------------------------
    if not manifest_rows:
        logger.error("S4A: No outputs produced; manifest would be empty.")
        raise SystemExit(1)

    cols = [
        "method",
        "id",
        "path_gt",
        "path_deg",
        "path_restored",
        "degradation",
        "split",
        "restored_w",
        "restored_h",
    ]
    pd.DataFrame(manifest_rows, columns=cols).to_csv(s4_manifest_path, index=False)
    logger.info("S4A: Wrote manifest: %s (rows=%d)", s4_manifest_path, len(manifest_rows))

    # Sanity: check that all restored files exist
    restored_paths = [r["path_restored"] for r in manifest_rows]
    exist_n = count_existing(restored_paths)
    if exist_n != len(restored_paths):
        logger.error(
            "S4A: Sanity check failed: manifest rows=%d, existing restored files=%d.",
            len(restored_paths),
            exist_n,
        )
        raise SystemExit(1)

    logger.info(
        "S4A: Sanity check passed. All %d restored files present.",
        exist_n,
    )
    logger.info("S4A: GFPGAN inference stage completed successfully.")
