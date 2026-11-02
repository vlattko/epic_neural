"""Unit tests for Grounded Empirical Investigation and Anti-Hallucination Framework.

Guarantees:
- Zero mock or synthetic data.
- Strict provenance against validation_1000pts_raw_comparison.csv.
- Verified physical failure regimes and hybrid dispatcher metrics.
"""

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
RAW_CSV_PATH = Path(
    r"D:\Soil Water Nexus - HORIZON-MISS-2025-05-SOIL-04"
    r"\kickoff\validation_results\validation_1000pts_raw_comparison.csv"
)

from src.analysis.surrogate_failure_physics import (
    load_ground_truth,
    calculate_metrics,
    analyze_failure_regimes,
    design_physics_informed_hybrid_dispatcher,
)
from src.analysis.pilot_drava_basin import analyze_pilot_drava
from src.agents.grounded_investigation import FactCheckAuditEngine


def test_raw_ground_truth_integrity():
    """Verify that empirical ground truth exists and has expected shape and schemas."""
    df = load_ground_truth(RAW_CSV_PATH)
    assert len(df) == 5000
    assert df["parcel_id"].nunique() == 1000
    assert set(df["scenario"].unique()) == {
        "historical_baseline",
        "ssp126_conservation",
        "ssp245_conventional",
        "ssp245_precision_water",
        "ssp585_conventional",
    }
    assert (~df["is_test"]).sum() == 3500
    assert (df["is_test"]).sum() == 1500


def test_surrogate_failure_regimes_metrics():
    """Verify exact quantification of the three physical failure regimes."""
    df = load_ground_truth(RAW_CSV_PATH)
    regimes = analyze_failure_regimes(df)

    # Regime 1: Zero-Inflation Non-Negativity Violation
    r1 = regimes["regime_1_zero_inflation"]
    assert r1["dryland_sample_size"] == 4000
    assert r1["gpu_negative_irrigation_count"] == 1614
    assert r1["gpu_negative_irrigation_min_val_mm"] == pytest.approx(-4.5364, abs=1e-3)
    assert r1["gpu_phantom_irrigation_positive_count"] == 2279
    assert r1["dryland_r2"] == 0.0

    # Regime 2: Alpine / Montane Under SSP5-8.5
    r2 = regimes["regime_2_alpine_extremes"]
    assert r2["alpine_ssp585_count"] == 69
    assert r2["soc_delta_r2"] == pytest.approx(0.1659, abs=1e-3)
    assert r2["water_stress_days_r2"] == pytest.approx(0.3075, abs=1e-3)

    # Regime 3: Topographic Slope Erosion
    r3 = regimes["regime_3_topographic_erosion"]["by_slope_band"]["steep_8_to_15"]
    assert r3["count"] == 315
    assert r3["max_abs_error_t_ha"] == pytest.approx(37.35, abs=1e-2)
    assert r3["mae_t_ha"] == pytest.approx(5.24, abs=1e-2)


def test_hybrid_dispatcher_guardrail():
    """Verify workload allocation and error reductions of the Hybrid Dispatcher."""
    df = load_ground_truth(RAW_CSV_PATH)
    hybrid = design_physics_informed_hybrid_dispatcher(df)

    assert hybrid["total_evaluations"] == 5000
    assert hybrid["surrogate_safe_operating_space_count"] == 4655
    assert hybrid["surrogate_safe_operating_space_pct"] == 93.1
    assert hybrid["fortran_dispatched_count"] == 345
    assert hybrid["fortran_dispatched_pct"] == 6.9

    # USLE erosion improvement
    ero = hybrid["performance_improvement"]["usle_erosion"]
    assert ero["raw_surrogate_max_err_t_ha"] == pytest.approx(37.35, abs=1e-2)
    assert ero["hybrid_guardrail_max_err_t_ha"] == pytest.approx(5.31, abs=1e-2)
    assert ero["rmse_reduction_pct"] == pytest.approx(68.95, abs=1e-2)

    # Irrigation improvement
    irr = hybrid["performance_improvement"]["irrigation"]
    assert irr["negative_irrigation_eliminated"] is True


def test_pilot_drava_grounded(tmp_path):
    """Verify empirical analysis of the Croatian Drava River basin pilot site."""
    res = analyze_pilot_drava(tmp_path)
    assert res["num_arkod_parcels"] == 54
    assert res["total_area_ha"] == pytest.approx(449.71, abs=1e-2)
    assert res["hydrological_stress"]["total_days_evaluated"] == 184
    assert res["hydrological_stress"]["water_deficit_days_swdi_negative"] == 144
    assert res["hydrological_stress"]["severe_deficit_days_swdi_under_minus_two"] == 101


def test_fact_check_audit_engine(tmp_path):
    """Verify that FactCheckAuditEngine catches mock/random tokens and verifies grounded numbers."""
    dummy_json = tmp_path / "grounded_hybrid_dispatcher.json"
    dummy_json.write_text(json.dumps({
        "surrogate_safe_operating_space_pct": 93.1,
        "fortran_dispatched_pct": 6.9,
    }), encoding="utf-8")

    engine = FactCheckAuditEngine([dummy_json])

    # Text containing forbidden mock token
    bad_text = "We used np.random to simulate 1,284,769 parcels with 93.1% safe space."
    res_bad = engine.audit_text(bad_text)
    assert res_bad["all_verified"] is False
    assert res_bad["mock_or_random_detected"] is True

    # Valid grounded text
    good_text = "The system achieved 93.1% surrogate safe space and 6.9% Fortran fallback."
    res_good = engine.audit_text(good_text)
    assert res_good["all_verified"] is True
    assert res_good["mock_or_random_detected"] is False
    assert res_good["passed_checks"] >= 2
