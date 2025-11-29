# src/s1_data/utils/validators.py

from typing import Dict, Iterable, Set

import pandas as pd

from src.utils.logging import get_logger


logger = get_logger("S1_VAL")


def _id_column(df: pd.DataFrame) -> str:
    """Return the column name used as ID (first column)."""
    return str(df.columns[0])


def validate_csv_schemas(metadata_frames: Dict[str, pd.DataFrame]) -> None:
    """Lightweight structural validation of metadata CSVs.

    Checks:
    - At least two columns per CSV (ID + one or more attributes).
    - No duplicate column names.
    Logs column names for visibility.
    """
    for key, df in metadata_frames.items():
        cols = list(df.columns)

        if len(cols) < 2:
            logger.error(
                "S1: CSV '%s' appears malformed: expected >=2 columns, found %d.",
                key,
                len(cols),
            )
            raise SystemExit(1)

        if len(cols) != len(set(cols)):
            logger.error(
                "S1: CSV '%s' has duplicate column names: %s.",
                key,
                cols,
            )
            raise SystemExit(1)

        logger.info("S1: CSV '%s' columns: %s", key, ", ".join(map(str, cols)))


def _ids_from_frame(df: pd.DataFrame) -> Set[str]:
    """Extract the ID set from a metadata frame (first column as string)."""
    id_col = _id_column(df)
    return set(df[id_col].astype(str))


def validate_id_consistency(
    image_filenames: Set[str],
    metadata_frames: Dict[str, pd.DataFrame],
) -> None:
    """Check ID consistency between image filenames and each metadata CSV.

    Assumes the first column of each DF contains the image filename or ID.
    """
    if not image_filenames:
        logger.error("S1: No image filenames provided for ID consistency check.")
        raise SystemExit(1)

    for key, df in metadata_frames.items():
        csv_ids = _ids_from_frame(df)

        missing_in_images = csv_ids - image_filenames
        missing_in_csv = image_filenames - csv_ids

        if missing_in_images:
            sample = sorted(list(missing_in_images))[:10]
            logger.error(
                "S1: CSV '%s' references %d IDs not found in images. Sample: %s",
                key,
                len(missing_in_images),
                ", ".join(sample),
            )
            raise SystemExit(1)

        # For strictness, enforce that all images have metadata everywhere.
        if missing_in_csv:
            sample = sorted(list(missing_in_csv))[:10]
            logger.error(
                "S1: %d images are missing from CSV '%s'. Sample: %s",
                len(missing_in_csv),
                key,
                ", ".join(sample),
            )
            raise SystemExit(1)

        logger.info(
            "S1: ID consistency OK for CSV '%s' (%d IDs).",
            key,
            len(csv_ids),
        )


def validate_partitions(partition_df: pd.DataFrame) -> None:
    """Validate the evaluation partition CSV.

    Assumes:
    - first column is image ID
    - second column is integer partition label in {0,1,2}
    """
    cols = list(partition_df.columns)
    if len(cols) < 2:
        logger.error(
            "S1: Partition CSV must have at least 2 columns (id, partition)."
        )
        raise SystemExit(1)

    id_col = cols[0]
    part_col = cols[1]

    labels = partition_df[part_col]
    unique_labels = set(labels.unique())

    allowed = {0, 1, 2}
    if not unique_labels.issubset(allowed):
        logger.error(
            "S1: Partition CSV has invalid labels %s (allowed: %s).",
            unique_labels,
            allowed,
        )
        raise SystemExit(1)

    counts = labels.value_counts().to_dict()
    logger.info("S1: Partition label distribution: %s", counts)

    # Optional: check for duplicate IDs
    dup_ids = partition_df[id_col].astype(str).duplicated()
    num_dups = int(dup_ids.sum())
    if num_dups > 0:
        logger.error(
            "S1: Partition CSV has %d duplicate IDs in column '%s'.",
            num_dups,
            id_col,
        )
        raise SystemExit(1)


def log_dataset_summary(
    image_filenames: Set[str],
    metadata_frames: Dict[str, pd.DataFrame],
) -> None:
    """Log a concise summary of the dataset state after all checks."""
    logger.info(
        "S1: Dataset summary — images: %d, metadata tables: %d.",
        len(image_filenames),
        len(metadata_frames),
    )

    # Intersection size across all metadata tables
    id_sets = [_ids_from_frame(df) for df in metadata_frames.values()]
    if not id_sets:
        logger.error("S1: No metadata frames available for summary.")
        raise SystemExit(1)

    intersection = set.intersection(*id_sets)
    logger.info(
        "S1: IDs present in all metadata CSVs: %d.",
        len(intersection),
    )
