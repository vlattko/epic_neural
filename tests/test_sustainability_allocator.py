"""Unit tests for multi-objective Pareto sustainability allocator."""

import sys
from pathlib import Path
import pytest
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optimization.sustainability_allocator import (
    ParetoAllocator,
    OptimizationResult,
    ScenarioMetrics,
    NOVEL_CROPS,
    LAND_USE_CLASSES,
    main,
)
from src.data.soil_loader import load_soil_data


@pytest.fixture(scope="session")
def soil_data():
    return load_soil_data()


@pytest.fixture(scope="session")
def allocator(soil_data):
    return ParetoAllocator(soil_data)


def test_schema_and_constants():
    assert "sorghum_grain" in NOVEL_CROPS
    assert "pearl_millet" in NOVEL_CROPS
    assert "triticale_winter" in NOVEL_CROPS
    assert "perennial_legume_mix" in NOVEL_CROPS
    assert "conventional_cropland" in LAND_USE_CLASSES
    assert "conservation_cropland" in LAND_USE_CLASSES
    assert "permanent_grassland" in LAND_USE_CLASSES


def test_evaluate_scenario(allocator):
    metrics = allocator.evaluate_scenario("ssp245_conventional")
    assert isinstance(metrics, ScenarioMetrics)
    assert metrics.n_parcels == 1000
    assert metrics.total_area_ha > 0
    assert metrics.total_production_t > 0
    assert metrics.total_erosion_t > 0
    assert metrics.total_water_deficit_m3 > 0
    assert metrics.mean_water_stress_days > 0


def test_invalid_scenario_raises(allocator):
    with pytest.raises(ValueError, match="not found"):
        allocator.evaluate_scenario("non_existent_scenario_123")


def test_optimize_reallocation_preserves_production(allocator):
    res = allocator.optimize_reallocation(
        scenario="ssp245_conventional",
        min_production_ratio=1.00,
    )
    assert isinstance(res, OptimizationResult)
    # Production must be preserved at >= 100% of baseline
    assert res.delta_production_pct >= -1e-5
    assert res.optimized_metrics.total_production_t >= res.baseline_metrics.total_production_t

    # Soil erosion must be reduced by at least 60%
    assert res.delta_erosion_pct < -60.0
    assert res.optimized_metrics.total_erosion_t < res.baseline_metrics.total_erosion_t

    # Water deficit must be reduced by at least 30%
    assert res.delta_water_deficit_pct < -30.0
    assert res.optimized_metrics.total_water_deficit_m3 < res.baseline_metrics.total_water_deficit_m3

    # Soil organic carbon must have net gain
    assert res.delta_soc_t_c > 0.0

    # Pareto dominance proven
    assert res.is_pareto_dominant is True


def test_slope_protection_rule(allocator):
    res = allocator.optimize_reallocation(
        scenario="ssp245_conventional",
        max_slope_annual_cropland_deg=5.0,
    )
    df = res.reallocated_df
    steep_parcels = df[df["slope_deg"] >= 8.0]
    # Steep parcels must be retired from conventional annual row crops
    assert (steep_parcels["assigned_land_use"] == "permanent_grassland").all()


def test_novel_crop_distribution(allocator):
    res = allocator.optimize_reallocation(
        scenario="ssp245_conventional",
        enable_novel_crops=True,
    )
    df = res.reallocated_df
    # Drought-stressed parcels should adopt novel crops
    assert "pearl_millet" in df["assigned_crop"].values
    assert "sorghum_grain" in df["assigned_crop"].values


def test_pareto_frontier(allocator):
    frontier = allocator.compute_pareto_frontier(
        scenario="ssp245_conventional",
        min_ratios=(0.95, 1.00, 1.05),
    )
    assert isinstance(frontier, pd.DataFrame)
    assert len(frontier) == 3
    assert "production_target_ratio" in frontier.columns
    assert "delta_production_pct" in frontier.columns
    assert "delta_erosion_pct" in frontier.columns
    assert "delta_water_deficit_pct" in frontier.columns
    # Check monotonic production scaling
    prods = frontier["production_achieved_t"].tolist()
    assert prods[0] <= prods[1] <= prods[2]


def test_cli_execution():
    code = main(["--scenario", "ssp126_conservation", "--min-prod", "0.95"])
    assert code == 0
