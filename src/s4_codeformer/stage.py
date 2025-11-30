import os
from collections import defaultdict

import pandas as pd

from src.utils.logging import get_logger
from src.s4_codeformer.utils.io import (
    ensure_dir,
    read_manifest_csv,
    load_image_rgb,
    save_image_jpeg,
    count_existing,
)
from src.s4_codeformer.utils.model import load_codeformer, run_codeformer


def run(config):
    logger = get_logger("S4_CF")
    logger.info("S4B: CodeFormer inference stage started.")

    # ------------------------------------------------------------------
    # Load S3 manifest
    # ------------------------------------------------------------------
    s3_manifest = os.path.join("results", "logs", "s3_degrade_manifest.csv")
    df = read_manifest_csv(s3_manifest)

    total_rows = len(df)
    if total_rows == 0:
        logger.error("S4B: S3 manifest is empty: %s", s3_manifest)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Output locations
    # ------------------------------------------------------------------
    outputs_root = os.path.join("results", "outputs", "codeformer")
    logs_root = os.path.join("results", "logs")
    ensure_dir(outputs_root)
    ensure_dir(logs_root)

    s4_manifest_path = os.path.join(logs_root, "s4_codeformer_manifest.csv")

    # ------------------------------------------------------------------
    # Load model + fidelity sweep settings
    # ------------------------------------------------------------------
    cfg_cf = config.get("upstreams", {}).get("codeformer", {})
    ckpt_name = cfg_cf.get("default_checkpoint", "codeformer-v0.1.0")

    fidelity_grid = (
        config.get("experiments", {})
        .get("matrix", {})
        .get("codeformer_fidelity_w", [0.5])
    )

    model = load_codeformer(ckpt_name=ckpt_name)
    if model is None:
        logger.error("S4B: CodeFormer dependencies missing or checkpoint not found.")
        raise SystemExit(1)

    logger.info("S4B: Fidelity sweep will run on w values: %s", fidelity_grid)

    # ------------------------------------------------------------------
    # Group by degradation preset
    # ------------------------------------------------------------------
    by_preset = defaultdict(list)
    for row in df.itertuples(index=False):
        by_preset[getattr(row, "degradation")].append(row)

    logger.info(
        "S4B: Inference will run over %d presets. Total samples: %d.",
        len(by_preset),
        total_rows,
    )

    # ------------------------------------------------------------------
    # Per-preset × fidelity inference
    # ------------------------------------------------------------------
    manifest_rows = []
    processed_total = 0
    target_total = total_rows * max(len(fidelity_grid), 1)

    for preset, rows in by_preset.items():
        logger.info("S4B: Processing preset '%s' with %d images.", preset, len(rows))

        for w in fidelity_grid:
            out_dir = os.path.join(outputs_root, preset, f"w_{w}", "imgs")
            ensure_dir(out_dir)

            logger.info(
                "S4B: Preset '%s' | fidelity w=%s -> %s",
                preset,
                w,
                out_dir,
            )

            for row in rows:
                image_id = getattr(row, "id")
                path_gt = getattr(row, "path_gt")
                path_deg = getattr(row, "path_deg")
                split = int(getattr(row, "split"))

                # Load degraded input
                try:
                    img_in = load_image_rgb(path_deg)
                except Exception as e:
                    logger.error("S4B: Failed to load degraded '%s': %s", path_deg, e)
                    continue

                # Model inference
                try:
                    restored = run_codeformer(model, img_in, fidelity=w)
                except Exception as e:
                    logger.error("S4B: CodeFormer failed on '%s': %s", path_deg, e)
                    continue

                # Save output
                path_restored = os.path.join(out_dir, image_id)
                try:
                    save_image_jpeg(restored, path_restored, quality=95)
                except Exception as e:
                    logger.error("S4B: Save failed '%s': %s", path_restored, e)
                    continue

                # Record manifest row
                manifest_rows.append(
                    {
                        "method": "codeformer",
                        "id": image_id,
                        "path_gt": path_gt,
                        "path_deg": path_deg,
                        "path_restored": path_restored,
                        "degradation": preset,
                        "split": split,
                        "w": w,
                        "restored_w": restored.width,
                        "restored_h": restored.height,
                    }
                )

                processed_total += 1
                if processed_total % 1000 == 0:
                    logger.info(
                        "S4B: Processed %d / %d total samples...",
                        processed_total,
                        target_total,
                    )

        logger.info("S4B: Finished preset '%s'.", preset)

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    if not manifest_rows:
        logger.error("S4B: No outputs produced; manifest would be empty.")
        raise SystemExit(1)

    cols = [
        "method",
        "id",
        "path_gt",
        "path_deg",
        "path_restored",
        "degradation",
        "split",
        "w",
        "restored_w",
        "restored_h",
    ]
    pd.DataFrame(manifest_rows, columns=cols).to_csv(s4_manifest_path, index=False)
    logger.info("S4B: Wrote manifest: %s (rows=%d)", s4_manifest_path, len(manifest_rows))

    # ------------------------------------------------------------------
    # Sanity: check outputs exist
    # ------------------------------------------------------------------
    restored_paths = [r["path_restored"] for r in manifest_rows]
    exist_n = count_existing(restored_paths)
    if exist_n != len(restored_paths):
        logger.error(
            "S4B: Sanity check failed: manifest rows=%d, existing restored files=%d.",
            len(restored_paths),
            exist_n,
        )
        raise SystemExit(1)

    logger.info("S4B: Sanity check passed. All %d restored files present.", exist_n)
    logger.info("S4B: CodeFormer inference stage completed successfully.")
