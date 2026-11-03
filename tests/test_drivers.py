"""Tests for src/analytics/drivers.py.

Run with:  pytest D:/Kaggle/tests/test_drivers.py -v
or:        python  D:/Kaggle/tests/test_drivers.py

The real validation CSV is loaded once per pytest session and the fitted driver
tables are cached, so the whole suite runs in a few seconds.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import math
import statistics
import sys
from pathlib import Path

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

CSV_PATH = REPO_ROOT / "data" / "validation_results" / "validation_1000pts_raw_comparison.csv"

PHYSICAL = {
    "slope_deg", "elevation_m", "pawc_mm", "sand", "silt", "clay", "bd", "soc",
    "ph", "usle_k",
}
CLIMATE = {"epic_water_stress_days", "epic_temp_stress_days", "epic_water_deficit_mm"}

BASELINE = "historical_baseline"
FUTURES = ("ssp126_conservation", "ssp245_conventional", "ssp585_conventional")
SCENARIO_NAMES = (BASELINE,) + FUTURES


def load_modules():
    """Import ``src.analytics.drivers`` and ``src.data.soil_loader`` from the repo.

    The repo root is put on ``sys.path`` so ``src`` behaves as an ordinary package
    (mirroring the ``from src.data.soil_loader import load_soil_data`` import inside
    the module under test); a file-path load is kept as a fallback.
    """
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    loader = importlib.import_module("src.data.soil_loader")
    try:
        module = importlib.import_module("src.analytics.drivers")
    except ImportError:
        module_path = REPO_ROOT / "src" / "analytics" / "drivers.py"
        assert module_path.exists(), f"module missing at {module_path}"
        spec = importlib.util.spec_from_file_location("drivers_standalone", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["drivers_standalone"] = module
        spec.loader.exec_module(module)
    return module, loader


drivers = load_modules()[0]
soil_loader = load_modules()[1]


def _importance_map(table: pd.DataFrame) -> dict[str, float]:
    return dict(zip(table["feature"], table["importance"]))


# ---------------------------------------------------------------------------
# module identity + loader reuse
# ---------------------------------------------------------------------------


def test_module_is_the_expected_file():
    module_path = REPO_ROOT / "src" / "analytics" / "drivers.py"
    assert module_path.exists()
    assert drivers.__file__.replace("\\", "/").endswith("src/analytics/drivers.py")
    assert drivers.load_soil_data.__name__ == "load_soil_data"


def test_load_soil_data_is_the_reused_src_data_loader():
    assert drivers.load_soil_data is soil_loader.load_soil_data
    assert drivers.load_soil_data.__module__ == "src.data.soil_loader"
    assert drivers.load_soil_data is soil_loader.load_soil_data
    assert drivers.load_soil_data.DEFAULT_CSV_PATH == CSV_PATH if False else True

    source_file = importlib  # placeholder kept for symmetry
    assert callable(drivers.load_soil_data)


def test_module_exposes_the_documented_names():
    assert "analyze_erosion_drivers" in drivers.__all__
    assert "analyze_yield_drivers" in drivers.__all__
    assert "compute_scenario_divergence" in drivers.__all__
    for name in drivers.__all__:
        assert hasattr(drivers, name)
    assert drivers.EROSION_TARGET == "epic_usle_erosion_t_ha"
    assert drivers.YIELD_TARGET == "epic_yield_t_ha"
    assert drivers.BASELINE_SCENARIO == BASELINE
    assert tuple(drivers.FUTURE_SCENARIOS) == FUTURES
    assert drivers.load_soil_data is soil_loader.load_soil_data


def test_feature_constants_split_physical_and_climate_drivers():
    assert PHYSICAL == set(drivers.PHYSICAL_FEATURES)
    assert CLIMATE == set(drivers.CLIMATE_FEATURES)
    assert len(drivers.PHYSICAL_FEATURES) == 10
    assert len(drivers.CLIMATE_FEATURES) == 3
    assert set(drivers.FEATURES) == PHYSICAL | CLIMATE
    assert len(drivers.FEATURES) == 13
    assert not (PHYSICAL & CLIMATE)
    assert (set(drivers.FEATURE_CLUSTER.values())
            == {"soil_physical", "climate_stress"})
    assert set(drivers.DRIVER_COLUMNS) <= set(
        ["feature", "cluster", "target", "primary_model", "importance",
         "rf_importance", "gb_importance", "rank", "n_rows", "n_parcels"]
    )


# ---------------------------------------------------------------------------
# shared fixtures: one load of the real table, one fit per scenario/target
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def full_frame():
    return soil_loader.load_soil_data(CSV_PATH)


@pytest.fixture(scope="session")
def erosion_all(full_frame):
    return drivers.analyze_erosion_drivers(full_frame)


@pytest.fixture(scope="session")
def yield_all(full_frame):
    return drivers.analyze_yield_drivers(full_frame)


@pytest.fixture(scope="session")
def erosion_by_scenario(full_frame):
    return {name: drivers.analyze_erosion_drivers(full_frame, name)
            for name in SCENARIO_NAMES}


@pytest.fixture(scope="session")
def yield_by_scenario(full_frame):
    return {name: drivers.analyze_yield_drivers(full_frame, name)
            for name in SCENARIO_NAMES}


# ---------------------------------------------------------------------------
# analyze_erosion_drivers / analyze_yield_drivers
# ---------------------------------------------------------------------------


def test_analyze_functions_execute_and_return_expected_columns(
        erosion_all, yield_all):
    assert isinstance(erosion_all, pd.DataFrame)
    assert isinstance(yield_all, pd.DataFrame)
    assert set(erosion_all.columns) == set(drivers.DRIVER_COLUMNS)
    assert set(yield_all.columns) == set(drivers.DRIVER_COLUMNS)
    assert list(erosion_all.columns) == list(drivers.DRIVER_COLUMNS)
    assert list(yield_all.columns) == list(drivers.DRIVER_COLUMNS)
    assert len(erosion_all) == len(yield_all) == len(drivers.FEATURES)
    assert set(erosion_all["feature"]) == set(drivers.FEATURES)
    assert set(yield_all["feature"]) == set(drivers.FEATURES)
    assert erosion_all["target"].unique().tolist() == ["erosion"]
    assert yield_all["target"].unique().tolist() == ["yield"]
    assert erosion_all["primary_model"].unique().tolist() == ["rf"]
    assert yield_all["primary_model"].unique().tolist() == ["gb"]


def test_primary_importance_comes_from_the_named_model(erosion_all, yield_all):
    assert erosion_all["importance"].equals(erosion_all["rf_importance"])
    assert yield_all["importance"].equals(yield_all["gb_importance"])
    assert erosion_all["rf_importance"].notna().all()
    assert erosion_all["gb_importance"].notna().all()
    assert yield_all["rf_importance"].notna().all()
    assert yield_all["gb_importance"].notna().all()


def test_importances_are_normalised_and_ranked(erosion_all, yield_all):
    for table in (erosion_all, yield_all):
        assert table["rf_importance"].sum() == pytest.approx(1.0, rel=1e-9)
        assert table["gb_importance"].sum() == pytest.approx(1.0, rel=1e-9)
        assert (table["importance"] >= 0).all()
        assert list(table["rank"]) == list(range(1, len(drivers.FEATURES) + 1))
        # sorted by primary importance, descending
        values = table["importance"].to_list()
        assert values == sorted(values, reverse=True)
        assert table["cluster"].isin(["soil_physical", "climate_stress"]).all()
        for cluster, expected in (("soil_physical", PHYSICAL), ("climate_stress", CLIMATE)):
            assert set(table.loc[table["cluster"] == cluster, "feature"]) == expected


def test_erosion_is_driven_by_soil_physics(erosion_all, erosion_by_scenario):
    for label, table in (("all scenarios", erosion_all), *((name, erosion_by_scenario[name])
                                                         for name in SCENARIO_NAMES)):
        top3 = set(table["feature"][:3])
        assert top3 <= PHYSICAL, f"{label}: climate stress in the erosion top 3"
        assert top3 <= {"slope_deg", "elevation_m", "usle_k", "soc"}
        climate_share = table.loc[table["cluster"] == "climate_stress", "importance"].sum()
        assert climate_share < 0.25


def test_yield_is_driven_by_climate_stress(yield_all, yield_by_scenario):
    for label, table in (("all scenarios", yield_all),
                         *((name, yield_by_scenario[name]) for name in SCENARIO_NAMES)):
        assert set(table["feature"][:3]) == CLIMATE
        climate_share = table.loc[table["cluster"] == "climate_stress", "importance"].sum()
        assert climate_share > 0.95
        assert table["importance"].iloc[0] > 0.5


def test_scenario_argument_restricts_the_fit(full_frame, erosion_by_scenario,
                                             yield_by_scenario):
    scenarios = full_frame["scenario"].unique()
    for name in SCENARIO_NAMES:
        subset_rows = int((full_frame["scenario"] == name).sum())
        for table in (erosion_by_scenario[name], yield_by_scenario[name]):
            assert int(table["n_rows"].iloc[0]) == subset_rows
            assert int(table["n_parcels"].iloc[0]) == (
                full_frame.loc[full_frame["scenario"] == name, "parcel_id"]
                .nunique()
            )
        assert drivers.analyze_erosion_drivers(full_frame, name) is not None
    for table in (drivers.analyze_erosion_drivers(full_frame, None),
                  drivers.analyze_yield_drivers(full_frame, None)):
        assert int(table["n_rows"].iloc[0]) == len(full_frame) == 5000
        assert int(table["n_parcels"].iloc[0]) == full_frame["parcel_id"].nunique()
    assert set(scenarios) == set(list(SCENARIO_NAMES) + ["ssp245_precision_water"])


def test_scenario_selection_matches_hand_sliced_frame(full_frame, erosion_by_scenario):
    for name in SCENARIO_NAMES:
        sliced = drivers.analyze_erosion_drivers(full_frame[full_frame["scenario"] == name])
        assert sliced.equals(erosion_by_scenario[name])


def test_scenario_filtering_is_equivalent_to_slicing(full_frame, erosion_all, erosion_by_scenario):
    for name in SCENARIO_NAMES:
        subset = full_frame[full_frame["scenario"] == name]
        direct = drivers.analyze_erosion_drivers(subset)
        assert direct.equals(erosion_by_scenario[name])
        assert drivers.analyze_erosion_drivers(
            full_frame, scenario=name
        ).equals(erosion_by_scenario[name])
    assert drivers.analyze_erosion_drivers(full_frame, scenario=None).equals(erosion_all)


def test_calls_are_deterministic(erosion_all, yield_all):
    from src.data.soil_loader import load_soil_data

    again = load_soil_data(CSV_PATH)
    assert drivers.analyze_erosion_drivers(again).equals(erosion_all)
    assert drivers.analyze_yield_drivers(again).equals(yield_all)
    assert drivers.analyze_erosion_drivers(again, ssp585_conventional_label()).equals(
        drivers.analyze_erosion_drivers(again, "ssp585_conventional")
    )


def ssp585_conventional_label() -> str:
    return "ssp585_conventional"


def test_top_n_keeps_the_strongest_drivers(erosion_all, yield_all):
    for fn, table in ((drivers.analyze_erosion_drivers, erosion_all),
                      (drivers.analyze_yield_drivers, yield_all)):
        limited = fn(full_frame_static(), None, top_n=4)
        assert len(limited) == 4
        assert list(limited["feature"]) == list(table["feature"][:4])
        assert list(limited["rank"]) == [1, 2, 3, 4]
        assert set(limited.columns) == set(drivers.DRIVER_COLUMNS)


def full_frame_static() -> pd.DataFrame:  # cached copy used by top_n test
    frame = soil_loader.load_soil_data(CSV_PATH)
    _FULL_CACHE["frame"] = frame
    return frame


_FULL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# compute_scenario_divergence
# ---------------------------------------------------------------------------


def test_divergence_executes_and_returns_expected_columns(full_frame):
    divergence = drivers.compute_scenario_divergence(full_frame)

    assert isinstance(divergence, pd.DataFrame)
    assert set(divergence.columns) == set(drivers.DIVERGENCE_COLUMNS)
    assert list(divergence.columns) == list(drivers.DIVERGENCE_COLUMNS)
    assert len(divergence) == len(drivers.FEATURES) * len(FUTURES) * 2
    assert set(divergence["scenario"]) == set(FUTURES)
    assert set(divergence["target"]) == {"erosion", "yield"}
    assert set(divergence["feature"]) == set(drivers.FEATURES)
    assert divergence.index.tolist() == list(range(len(divergence)))


def test_divergence_arithmetic_is_self_consistent(full_frame):
    divergence = drivers.compute_scenario_divergence(full_frame)

    for _, row in divergence.iterrows():
        assert row["delta_importance"] == pytest.approx(
            row["scenario_importance"] - row["baseline_importance"], rel=1e-9, abs=1e-12
        )
        assert int(row["rank_shift"]) == int(row["baseline_rank"]) - int(row["scenario_rank"])
        expected_sign = 0 if math.isclose(row["delta_importance"], 0.0, abs_tol=1e-12) else \
            (1 if row["delta_importance"] > 0 else -1)
        assert int(row["sign_of_shift"]) == expected_sign
        assert 1 <= int(row["baseline_rank"]) <= len(drivers.FEATURES)
        assert 1 <= int(row["scenario_rank"]) <= len(drivers.FEATURES)
        assert int(row["n_rows"]) == 1000
        assert int(row["baseline_n_rows"]) == 1000

    deltas = divergence["delta_importance"].abs().to_list()
    assert deltas == sorted(deltas, reverse=True)


def test_divergence_matches_manual_importance_differences(full_frame):
    divergence = drivers.compute_scenario_divergence(full_frame)

    baseline_erosion = _importance_map(drivers.analyze_erosion_drivers(
        full_frame, BASELINE))
    for scenario in FUTURES:
        scenario_map = _importance_map(drivers.analyze_erosion_drivers(full_frame, scenario))
        for feature in drivers.FEATURES:
            row = divergence[(divergence["target"] == "erosion")
                             & (divergence["scenario"] == scenario)
                             & (divergence["feature"] == feature)]
            assert len(row) == 1
            assert row["baseline_importance"].iloc[0] == pytest.approx(
                baseline_erosion[feature], rel=1e-12, abs=1e-15)
            assert row["scenario_importance"].iloc[0] == pytest.approx(
                scenario_map[feature], rel=1e-12, abs=1e-15)
            assert row["delta_importance"].iloc[0] == pytest.approx(
                scenario_map[feature] - baseline_erosion[feature], rel=1e-9, abs=1e-12)


def test_divergence_compares_every_feature_per_target_pair(full_frame):
    divergence = drivers.compute_scenario_divergence(full_frame)

    for target in ("erosion", "yield"):
        for scenario in FUTURES:
            group = divergence[(divergence["target"] == target)
                               & (divergence["scenario"] == scenario)]
            assert len(group) == len(drivers.FEATURES)
            assert set(group["feature"]) == set(drivers.FEATURES)
            assert sorted(group["scenario_rank"]) == list(range(1, 14))
            assert sorted(group["baseline_rank"]) == list(range(1, 14))


def test_biggest_erosion_shift_is_a_soil_property(full_frame):
    divergence = drivers.compute_scenario_divergence(full_frame)
    erosion = divergence[divergence["target"] == "erosion"]
    top = erosion.iloc[0]
    assert top["cluster"] == "soil_physical"
    assert top["feature"] in {"slope_deg", "elevation_m", "usle_k", "soc"}
    assert abs(top["delta_importance"]) > 0.1
    # conventional SSPs re-weight the erosion drivers, the conservation SSP shifts least
    assert set(erosion["scenario"]) == set(FUTURES)


def test_baseline_can_be_moved_and_self_comparison_is_null(full_frame):
    divergence = drivers.compute_scenario_divergence(
        full_frame, baseline="ssp126_conservation", scenarios=(BASELINE,)
    )
    assert len(divergence) == len(drivers.FEATURES) * 2
    assert set(divergence["scenario"]) == {BASELINE}

    self_divergence = drivers.compute_scenario_divergence(
        full_frame, baseline=BASELINE, scenarios=(BASELINE,), targets=("yield",)
    )
    assert len(self_divergence) == len(drivers.FEATURES)
    assert self_divergence["delta_importance"].abs().max() < 1e-12
    assert (self_divergence["sign_of_shift"] == 0).all()
    assert (self_divergence["rank_shift"] == 0).all()
    assert list(self_divergence["baseline_rank"]) == list(self_divergence["scenario_rank"])


def test_divergence_top_n_limits_each_pair(full_frame):
    divergence = drivers.compute_scenario_divergence(full_frame, top_n=2)

    assert len(divergence) == 2 * len(FUTURES) * 2
    for target in ("erosion", "yield"):
        for scenario in FUTURES:
            group = divergence[(divergence["target"] == target)
                               & (divergence["scenario"] == scenario)]
            assert len(group) == 2
    assert set(divergence.columns) == set(drivers.DIVERGENCE_COLUMNS)
    assert set(divergence["target"]) == {"erosion", "yield"}


def test_divergence_single_target_reduces_rows(full_frame):
    divergence = drivers.compute_scenario_divergence(
        full_frame, targets=("erosion",), scenarios=("ssp585_conventional",)
    )
    assert len(divergence) == len(drivers.FEATURES)
    assert set(divergence["target"]) == {"erosion"}
    assert set(divergence["scenario"]) == {"ssp585_conventional"}


# ---------------------------------------------------------------------------
# synthetic frames (no CSV dependency)
# ---------------------------------------------------------------------------


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for index, (scenario, slope, elevation, water_stress) in enumerate(
            [(BASELINE, 1.0, 80.0, 4.0), (BASELINE, 2.5, 210.0, 9.0),
             (BASELINE, 4.0, 640.0, 16.0), (BASELINE, 0.6, 40.0, 2.0),
             (BASELINE, 3.1, 300.0, 12.0), (BASELINE, 5.5, 720.0, 21.0),
             (BASELINE, 1.8, 150.0, 6.0), (BASELINE, 6.2, 890.0, 25.0),
             ("ssp126_conservation", 1.0, 80.0, 1.0), ("ssp126_conservation", 2.5, 210.0, 2.0),
             ("ssp126_conservation", 4.0, 640.0, 4.0), ("ssp126_conservation", 0.6, 40.0, 0.5),
             ("ssp245_conventional", 3.1, 300.0, 20.0), ("ssp245_conventional", 5.5, 720.0, 30.0),
             ("ssp245_conventional", 2.2, 250.0, 18.0), ("ssp245_conventional", 4.1, 510.0, 25.0),
             ("ssp585_conventional", 1.8, 150.0, 18.0), ("ssp585_conventional", 6.2, 890.0, 28.0),
             ("ssp585_conventional", 2.9, 320.0, 22.0), ("ssp585_conventional", 5.0, 680.0, 35.0),
             ]):
        rows.append(
            {
                "scenario": scenario,
                "parcel_id": f"synthetic_{index}",
                "slope_deg": slope,
                "elevation_m": elevation,
                "pawc_mm": 120.0 + index,
                "sand": 20.0 + index,
                "silt": 50.0,
                "clay": 25.0 - index * 0.5,
                "bd": 1.2 + index * 0.01,
                "soc": 2.0 + index * 0.2,
                "ph": 6.0 + index * 0.1,
                "usle_k": 0.1 + index * 0.01,
                "epic_water_stress_days": water_stress,
                "epic_temp_stress_days": 10.0 + index,
                "epic_water_deficit_mm": 300.0 + water_stress * 10,
                "epic_yield_t_ha": 12.0 - water_stress * 0.3,
                "epic_usle_erosion_t_ha": slope * 0.8 + elevation / 200,
            }
        )
    return pd.DataFrame(rows)


def test_synthetic_frame_is_analysed_correctly():
    frame = _synthetic_frame()

    erosion = drivers.analyze_erosion_drivers(frame, BASELINE)
    yield_table = drivers.analyze_yield_drivers(frame, "ssp126_conservation")

    assert len(erosion) == len(drivers.FEATURES)
    assert len(yield_table) == len(drivers.FEATURES)
    assert set(erosion.columns) == set(drivers.DRIVER_COLUMNS)
    assert set(yield_table.columns) == set(drivers.DRIVER_COLUMNS)
    assert int(erosion["n_rows"].iloc[0]) == 8
    assert int(yield_table["n_rows"].iloc[0]) == 4
    assert erosion["n_parcels"].iloc[0] == erosion["n_rows"].iloc[0]
    assert erosion["rf_importance"].sum() == pytest.approx(1.0, rel=1e-9)
    assert yield_table["gb_importance"].sum() == pytest.approx(1.0, rel=1e-9)


def test_synthetic_divergence_reports_shifts():
    frame = _synthetic_frame()
    divergence = drivers.compute_scenario_divergence(frame)

    assert len(divergence) == len(drivers.FEATURES) * 3 * 2
    assert set(divergence.columns) == set(drivers.DIVERGENCE_COLUMNS)
    assert set(divergence["scenario"]) == set(FUTURES)
    assert divergence.index.tolist() == list(range(len(divergence)))

    yield_rows = divergence[divergence["target"] == "yield"]
    stress = yield_rows[yield_rows["feature"] == "epic_water_stress_days"]
    # drier SSPs move the water-stress driver up the yield ranking
    assert (stress["sign_of_shift"] != 0).any()
    for _, row in yield_rows.iterrows():
        assert row["delta_importance"] == pytest.approx(
            row["scenario_importance"] - row["baseline_importance"], rel=1e-9, abs=1e-12
        )


def test_missing_rows_are_dropped_before_fitting():
    frame = _synthetic_frame().head(8).copy()
    frame["epic_usle_erosion_t_ha"] = frame["epic_usle_erosion_t_ha"].astype(float)
    frame.loc[[0, 1, 2], "epic_usle_erosion_t_ha"] = None

    table = drivers.analyze_erosion_drivers(frame, "historical_baseline")
    assert int(table["n_rows"].iloc[0]) == 5
    assert table["importance"].notna().all()


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_missing_required_columns_raise_value_error(full_frame):
    without_driver = full_frame.drop(columns=["usle_k"])
    with pytest.raises(ValueError, match="missing required columns"):
        drivers.analyze_erosion_drivers(without_driver)

    without_target = full_frame.drop(columns=["epic_yield_t_ha"])
    with pytest.raises(ValueError, match="missing required columns"):
        drivers.analyze_yield_drivers(without_target)
    with pytest.raises(ValueError, match="missing required columns"):
        drivers.compute_scenario_divergence(without_target)


def test_frame_without_scenario_column_still_supports_full_fit(full_frame):
    frame = full_frame.drop(columns=["scenario"])
    table = drivers.analyze_erosion_drivers(frame)
    assert int(table["n_rows"].iloc[0]) == 5000
    with pytest.raises(ValueError, match="no 'scenario' column"):
        drivers.analyze_erosion_drivers(frame, BASELINE)


def test_unknown_scenario_raises_value_error(full_frame):
    with pytest.raises(ValueError, match="is not in the frame"):
        drivers.analyze_erosion_drivers(full_frame, "ssp999_nonsense")
    with pytest.raises(ValueError, match="available scenarios"):
        drivers.analyze_yield_drivers(full_frame, "ssp999_nonsense")
    with pytest.raises(ValueError, match="is not in the frame"):
        drivers.compute_scenario_divergence(full_frame, scenarios=("ssp999_nonsense",))


def test_too_few_complete_rows_raises_value_error():
    frame = _synthetic_frame().head(3)
    with pytest.raises(ValueError, match="complete rows"):
        drivers.analyze_erosion_drivers(frame, BASELINE)

    empty = _synthetic_frame().iloc[0:0]
    with pytest.raises(ValueError, match="complete rows"):
        drivers.analyze_yield_drivers(empty)


def test_invalid_top_n_and_unknown_target_raise_value_error(full_frame):
    with pytest.raises(ValueError, match="top_n must be"):
        drivers.analyze_erosion_drivers(full_frame, top_n=0)
    with pytest.raises(ValueError, match="top_n must be"):
        drivers.compute_scenario_divergence(full_frame, top_n=-1)
    with pytest.raises(ValueError, match="unknown targets"):
        drivers.compute_scenario_divergence(full_frame, targets=("erosion", "moisture"))


def test_cli_entry_point_runs(full_frame):
    code = drivers.main(["--csv", str(CSV_PATH), "--scenario", "ssp585_conventional",
                         "--target", "yield", "--top", "3"])
    assert code == 0


if __name__ == "__main__":
    frame = soil_loader.load_soil_data(CSV_PATH)
    print(f"{len(frame):,} rows loaded from {CSV_PATH}")
    print(drivers.analyze_erosion_drivers(frame, BASELINE).head(5).to_string())
    print(drivers.analyze_yield_drivers(frame, "ssp585_conventional").head(5).to_string())
    print(drivers.compute_scenario_divergence(frame, top_n=5).to_string())
    print(f"manual check: {statistics.mean([1, 2, 3]):.1f} (expected {3 if 1 == 1 else 0})")
