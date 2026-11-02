"""Unit tests for continental-scale Pareto allocator."""

import sys
from pathlib import Path
import pytest
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optimization.continental_allocator import (
    ContinentalParetoAllocator,
    ContinentalReallocationResult,
    main,
)
from src.data.continental_loader import ContinentalDataLoader


@pytest.fixture(scope="session")
def continental_allocator():
    return ContinentalParetoAllocator()


def test_optimize_cohort_preserves_production(continental_allocator):
    # Optimize a cohort of 10,000 parcels
    loader = ContinentalDataLoader()
    df = loader.load_parcels(limit=10000, scenario="ssp245_conventional")
    res = continental_allocator.optimize_cohort(df, scenario="ssp245_conventional", min_production_ratio=1.00)

    assert isinstance(res, ContinentalReallocationResult)
    assert res.n_parcels == 10000
    assert res.total_area_ha > 0

    # Production preservation
    assert res.delta_production_pct >= -1e-5
    assert res.optimized_production_t >= res.baseline_production_t

    # Soil erosion slashed (> 70%)
    assert res.delta_erosion_pct < -70.0
    assert res.optimized_erosion_t < res.baseline_erosion_t

    # Water deficit reduction (> 30%)
    assert res.delta_water_deficit_pct < -30.0

    # Net SOC gain
    assert res.delta_soc_t_c > 0.0

    # Pareto dominant
    assert res.is_pareto_dominant is True


def test_critical_slope_transition(continental_allocator):
    loader = ContinentalDataLoader()
    df = loader.load_parcels(limit=2000, min_slope=8.0)
    res = continental_allocator.optimize_cohort(df, critical_slope_deg=8.0)
    # All steep parcels must be converted to permanent grassland
    assert "permanent_grassland" in res.land_use_area_ha
    assert res.land_use_area_ha["permanent_grassland"] == pytest.approx(res.total_area_ha, rel=1e-5)


def test_cli_execution():
    code = main(["--parcels", "500", "--scenario", "ssp245_conventional", "--min-prod", "1.00"])
    assert code == 0
