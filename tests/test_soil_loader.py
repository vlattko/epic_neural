"""Tests for src/data/soil_loader.py.

Run with:  pytest D:/Kaggle/tests/test_soil_loader.py -v
or:        python  D:/Kaggle/tests/test_soil_loader.py
"""

from __future__ import annotations

import importlib
import io
import statistics
import sys
from pathlib import Path

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

CSV_PATH = REPO_ROOT / "data" / "validation_results" / "validation_1000pts_raw_comparison.csv"

EXPECTED_SUMMARY_COLUMNS = {
    "n_rows",
    "n_parcels",
    "mean_yield_t_ha",
    "mean_erosion_t_ha",
}


def load_module():
    """Import soil_loader by path so the test works from any CWD."""
    module_path = REPO_ROOT / "src" / "data" / "soil_loader.py"
    assert module_path.exists(), f"module missing at {module_path}"

    sys.path.insert(0, str(module_path.parent))
    try:
        return importlib.import_module("soil_loader")
    finally:
        sys.path.pop(0)


soil_loader = load_module()


def test_module_is_the_expected_file():
    assert soil_loader.__file__.replace("\\", "/").endswith("src/data/soil_loader.py")
    assert soil_loader.DEFAULT_CSV_PATH == CSV_PATH


def test_load_returns_dataframe_with_expected_shape_and_columns():
    df = soil_loader.load_soil_data(CSV_PATH)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5000
    assert {"scenario", "parcel_id", "gpu_yield_t_ha", "err_yield_t_ha"} <= set(df.columns)
    for column in ("gpu_yield_t_ha", "gpu_usle_erosion_t_ha", "err_usle_erosion_t_ha"):
        assert pd.api.types.is_numeric_dtype(df[column])


def test_default_path_loads_the_same_frame():
    default = soil_loader.load_soil_data()
    explicit = soil_loader.load_soil_data(CSV_PATH)

    assert list(default.columns) == list(explicit.columns)
    assert len(default) == len(explicit) == 5000
    assert default.equals(explicit)


def test_summary_has_one_row_per_scenario():
    df = soil_loader.load_soil_data(CSV_PATH)
    summary = soil_loader.get_summary(df)

    scenarios = df["scenario"].unique()
    assert len(summary) == len(scenarios) == 5
    assert set(summary.index) == set(scenarios)
    assert EXPECTED_SUMMARY_COLUMNS <= set(summary.columns)
    assert summary["n_rows"].to_dict() == {s: 1000 for s in scenarios}
    assert int(summary["n_rows"].sum()) == len(df)
    assert summary["mean_yield_t_ha"].notna().all()
    assert summary["mean_erosion_t_ha"].notna().all()
    assert (summary["mean_yield_t_ha"] > 0).all()
    assert (summary["mean_erosion_t_ha"] >= 0).all()


def test_summary_means_match_manual_pandas_means():
    df = soil_loader.load_soil_data(CSV_PATH)
    summary = soil_loader.get_summary(df)

    for scenario in df["scenario"].unique():
        subset = df[df["scenario"] == scenario]
        assert summary.loc[scenario, "mean_yield_t_ha"] == pytest.approx(
            subset["gpu_yield_t_ha"].mean(), rel=1e-9
        )
        assert summary.loc[scenario, "mean_erosion_t_ha"] == pytest.approx(
            subset["gpu_usle_erosion_t_ha"].mean(), rel=1e-9
        )
        assert summary.loc[scenario, "n_parcels"] == subset["parcel_id"].nunique()


def test_summary_supports_other_group_and_value_columns():
    df = soil_loader.load_soil_data(CSV_PATH)

    by_ecotype = soil_loader.get_summary(df, group_col="ecotype")
    assert set(by_ecotype.index) == set(df["ecotype"].unique())
    assert EXPECTED_SUMMARY_COLUMNS <= set(by_ecotype.columns)

    epic = soil_loader.get_summary(
        df,
        yield_col="epic_yield_t_ha",
        erosion_col="epic_usle_erosion_t_ha",
    )
    scenario = "ssp585_conventional"
    subset = df[df["scenario"] == scenario]
    assert epic.loc[scenario, "mean_yield_t_ha"] == pytest.approx(
        subset["epic_yield_t_ha"].mean(), rel=1e-9
    )
    assert epic.loc[scenario, "mean_erosion_t_ha"] == pytest.approx(
        subset["epic_usle_erosion_t_ha"].mean(), rel=1e-9
    )


def test_synthetic_frame_is_summarised_correctly():
    text = (
        "status,parcel_id,scenario,ecotype,note\n"
        "SUCCESS,p1,historical_baseline,Atlantic,ignore\n"
        "SUCCESS,p2,ssp585_conventional,Atlantic,ignore\n"
        "SUCCESS,p1,ssp585_conventional,Atlantic,ignore\n"
    )
    df = pd.read_csv(io.StringIO(text))
    df["yield"] = [5.16, 9.10, 7.40]
    df["erosion"] = [0.20, 8.00, 9.00]

    summary = soil_loader.get_summary(df, yield_col="yield", erosion_col="erosion")

    assert len(summary) == 2
    assert set(summary.index) == {"historical_baseline", "ssp585_conventional"}
    assert summary.loc["ssp585_conventional", "mean_yield_t_ha"] == pytest.approx(
        statistics.mean([9.10, 7.40])
    )
    assert summary.loc["ssp585_conventional", "mean_erosion_t_ha"] == pytest.approx(
        statistics.mean([8.00, 9.00])
    )
    assert summary.loc["ssp585_conventional", "n_rows"] == 2
    assert summary.loc["ssp585_conventional", "n_parcels"] == 2
    assert "mean_err_yield_t_ha" not in summary.columns


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        soil_loader.load_soil_data(REPO_ROOT / "tests" / "definitely_missing.csv")


def test_columns_missing_from_csv_raise_value_error(tmp_path):
    broken = tmp_path / "broken_columns.csv"
    broken.write_text("status,parcel_id\nSUCCESS,p1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing expected columns"):
        soil_loader.load_soil_data(broken)


def test_summary_rejects_unknown_columns():
    df = soil_loader.load_soil_data(CSV_PATH)

    with pytest.raises(KeyError, match="not_found"):
        soil_loader.get_summary(df, yield_col="not_found")
    with pytest.raises(KeyError, match="not_found"):
        soil_loader.get_summary(df, erosion_col="not_found")
    with pytest.raises(KeyError, match="not_found"):
        soil_loader.get_summary(df, group_col="not_found")


if __name__ == "__main__":
    df = soil_loader.load_soil_data()
    summary = soil_loader.get_summary(df)
    print(f"{len(df):,} rows loaded from {soil_loader.DEFAULT_CSV_PATH}")
    print(f"{len(summary)} scenarios summarised")
    print(summary.to_string(float_format=lambda value: f"{value:,.4f}"))
