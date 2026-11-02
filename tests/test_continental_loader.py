"""Unit tests for continental data loader."""

import sys
from pathlib import Path
import pytest
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.continental_loader import (
    ContinentalDataLoader,
    ContinentalDatasetSummary,
    load_continental_sample,
    classify_elevation_tier,
    classify_soil_envirotype,
)


@pytest.fixture(scope="session")
def loader():
    return ContinentalDataLoader()


def test_continental_summary(loader):
    summary = loader.get_summary()
    assert isinstance(summary, ContinentalDatasetSummary)
    assert summary.total_parcels == 1284769
    assert summary.total_area_ha > 1000000.0  # > 1M ha
    assert summary.mean_slope_deg > 0.0
    assert summary.max_slope_deg > 60.0
    assert summary.steep_parcels_5deg > 200000
    assert summary.critical_slope_parcels_8deg > 100000
    assert summary.total_scenario_points == 1284769 * 5


def test_load_parcels_sampling(loader):
    sample = loader.load_parcels(limit=500, scenario="ssp245_conventional")
    assert isinstance(sample, pd.DataFrame)
    assert len(sample) == 500
    expected_cols = [
        "parcel_id", "area_ha", "slope_deg", "elevation_m", "pawc_mm", "soc", "usle_k",
        "epic_yield_t_ha", "epic_usle_erosion_t_ha", "epic_water_deficit_mm",
        "epic_water_stress_days", "epic_temp_stress_days", "epic_soc_delta_t_c_ha"
    ]
    for col in expected_cols:
        assert col in sample.columns


def test_slope_filtering(loader):
    steep = loader.load_parcels(limit=200, min_slope=8.0)
    assert (steep["slope_deg"] >= 8.0).all()


def test_hotspot_load(loader):
    nuts = loader.load_nuts2_hotspots()
    assert len(nuts) == 300
    assert "compound_vulnerability_index" in nuts.columns

    countries = loader.load_country_hotspots()
    assert len(countries) == 34


def test_classification_helpers():
    assert classify_elevation_tier(50.0) == "1. Lowland (<150m)"
    assert classify_elevation_tier(250.0) == "2. Rolling Hills (150-400m)"
    assert classify_elevation_tier(600.0) == "3. Sub-Montane (400-800m)"
    assert classify_elevation_tier(1200.0) == "4. Montane (>800m)"

    assert classify_soil_envirotype(200.0) == "High PAWC (>190mm, Loess/Chernozem)"
    assert classify_soil_envirotype(150.0) == "Medium PAWC (130-190mm, Luvisols)"
    assert classify_soil_envirotype(100.0) == "Low PAWC (<130mm, Shallow/Sandy)"
