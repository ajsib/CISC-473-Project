import os
import shutil
import random
from typing import Dict, Set, Optional
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("S1_PRUNE")

def prune_dataset(
    image_filenames: Set[str],
    metadata_frames: Dict[str, pd.DataFrame],
    csv_paths: Dict[str, str],
    image_dir: str,
    out_dir: str,
    sample_size: int,
    seed: Optional[int] = 1337,
) -> Set[str]:
    """
    Randomly subsample image IDs, copy corresponding images, and write pruned CSVs.
    """
    if sample_size >= len(image_filenames):
        logger.info("S1_PRUNE: Requested sample_size >= dataset size; skipping pruning.")
        return image_filenames

    rng = random.Random(seed)
    sampled_ids = set(rng.sample(sorted(image_filenames), sample_size))
    logger.info("S1_PRUNE: Sampled %d of %d images (seed=%s).", sample_size, len(image_filenames), seed)

    pruned_img_dir = os.path.join(out_dir, "img_align_celeba")
    os.makedirs(pruned_img_dir, exist_ok=True)

    # Copy sampled images to pruned dir
    for img_id in sampled_ids:
        src = os.path.join(image_dir, img_id)
        dst = os.path.join(pruned_img_dir, img_id)
        shutil.copyfile(src, dst)
    logger.info("S1_PRUNE: Copied %d images to '%s'.", len(sampled_ids), pruned_img_dir)

    # Write pruned CSVs
    for key, df in metadata_frames.items():
        id_col = str(df.columns[0])
        pruned_df = df[df[id_col].astype(str).isin(sampled_ids)]
        csv_out = os.path.join(out_dir, os.path.basename(csv_paths[key]))
        pruned_df.to_csv(csv_out, index=False)
        logger.info("S1_PRUNE: Wrote pruned CSV for '%s' (%d rows) to '%s'.", key, len(pruned_df), csv_out)

    logger.info("S1_PRUNE: Pruning completed: pruned dataset in '%s'.", out_dir)
    return sampled_ids
