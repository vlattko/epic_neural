"""Unit tests for continental driver analytics."""

import sys
from pathlib import Path
import pytest
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.continental_drivers import (
    ContinentalDriverAnalytics,
    ContinentalHotspotMetrics,
    main,
)


@pytest.fixture(scope="session")
def analytics():
    return ContinentalDriverAnalytics()


def test_nuts2_hotspots_evaluation(analytics):
    metrics = analytics.evaluate_nuts2_hotspots()
    assert isinstance(metrics, ContinentalHotspotMetrics)
    assert metrics.total_regions == 300
    assert len(metrics.most_vulnerable_regions) == 10
    assert len(metrics.highest_acceleration_regions) == 10
    assert metrics.mean_vulnerability_index > 0
    assert metrics.high_risk_region_count > 0

    # Mediterranean should have highest vulnerability among biomes
    assert metrics.ecotype_vulnerability_ranking["Mediterranean"] > metrics.ecotype_vulnerability_ranking["Boreal / Hemiboreal"]


def test_spatial_divergence_matrix(analytics):
    df = analytics.compute_spatial_divergence_matrix()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "compound_vulnerability_index" in df.columns


def test_cli_execution():
    code = main()
    assert code == 0
