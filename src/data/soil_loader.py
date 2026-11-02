"""Loader for the SOIL-04 validation comparison table.

Reads ``validation_1000pts_raw_comparison.csv`` (one row per parcel x scenario,
EPIC sequential output side by side with the GPU/numified output) into a
pandas DataFrame and provides a per-scenario summary of mean yield and mean
USLE erosion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV_PATH = Path(
    r"D:\Soil Water Nexus - HORIZON-MISS-2025-05-SOIL-04"
    r"\kickoff\validation_results\validation_1000pts_raw_comparison.csv"
)

REQUIRED_COLUMNS = (
    "scenario",
    "parcel_id",
    "epic_yield_t_ha",
    "epic_usle_erosion_t_ha",
    "gpu_yield_t_ha",
    "gpu_usle_erosion_t_ha",
)

# Columns averaged by get_summary(). Use the epic_* names for the sequential
# reference model instead (epic_yield_t_ha, epic_usle_erosion_t_ha).
YIELD_COLUMN = "gpu_yield_t_ha"
EROSION_COLUMN = "gpu_usle_erosion_t_ha"
YIELD_ERROR_COLUMN = "err_yield_t_ha"
EROSION_ERROR_COLUMN = "err_usle_erosion_t_ha"

# Kept as strings / booleans; everything else is coerced to a number.
NON_NUMERIC_COLUMNS = frozenset(
    {"status", "parcel_id", "scenario", "ecotype", "elevation_tier",
     "soil_envirotype", "hsg", "is_test"}
)


def load_soil_data(path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load the raw EPIC-vs-GPU validation comparison CSV.

    Args:
        path: CSV file to read. Defaults to :data:`DEFAULT_CSV_PATH`.

    Returns:
        DataFrame with one row per ``parcel_id``/``scenario`` pair, with
        numeric columns coerced to float.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a column from :data:`REQUIRED_COLUMNS` is missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No such validation CSV: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected columns: {', '.join(missing)}"
        )

    for column in df.columns:
        if column in NON_NUMERIC_COLUMNS:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def get_summary(
    df: pd.DataFrame,
    *,
    group_col: str = "scenario",
    yield_col: str = YIELD_COLUMN,
    erosion_col: str = EROSION_COLUMN,
) -> pd.DataFrame:
    """Mean yield and mean erosion aggregated by scenario.

    Args:
        df: Frame produced by :func:`load_soil_data`.
        group_col: Grouping column (``scenario`` by default; ``ecotype`` and
            ``soil_envirotype`` also work).
        yield_col: Column holding yield in t/ha.
        erosion_col: Column holding USLE erosion in t/ha.

    Returns:
        DataFrame indexed by the group value with ``n_rows``, ``n_parcels``,
        ``mean_yield_t_ha``, ``mean_erosion_t_ha``, and the mean comparison
        errors when those ``err_*`` columns exist in ``df``.

    Raises:
        KeyError: If ``group_col``, ``yield_col`` or ``erosion_col`` is absent.
    """
    for column in (group_col, yield_col, erosion_col):
        if column not in df.columns:
            raise KeyError(f"Column {column!r} not found in the loaded frame.")

    groups = df.groupby(group_col, observed=True)
    summary = pd.DataFrame({"n_rows": groups.size()})
    summary["n_parcels"] = groups["parcel_id"].nunique()

    summary["mean_yield_t_ha"] = groups[yield_col].mean()
    summary["mean_erosion_t_ha"] = groups[erosion_col].mean()

    for column, label in ((YIELD_ERROR_COLUMN, "mean_err_yield_t_ha"),
                          (EROSION_ERROR_COLUMN, "mean_err_erosion_t_ha")):
        if column in df.columns:
            summary[label] = groups[column].mean()

    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the per-scenario summary."""
    parser = argparse.ArgumentParser(description="Summarise a soil validation CSV")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="CSV to load")
    parser.add_argument("--group-col", default="scenario", help="Grouping column")
    args = parser.parse_args(argv)

    df = load_soil_data(args.csv)
    summary = get_summary(df, group_col=args.group_col)
    print(f"{len(df):,} rows loaded from {args.csv}")
    print(summary.to_string(float_format=lambda value: f"{value:,.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
