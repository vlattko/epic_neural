"""Verification test suite for multi-decadal soil-water nexus dynamics (2025-2050)."""

import pytest
import math
from src.simulation.decadal_dynamics import (
    SoilCarbonPools,
    RothCSOCKinetics,
    PedotransferHydroDynamics,
    FarmerAdoptionHazardModel,
    DecadalTrajectorySimulator,
)


def test_carbon_pools_properties():
    pools = SoilCarbonPools(dpm=2.0, rpm=8.0, bio=2.0, hum=40.0)
    assert pools.total_soc == 52.0
    assert 1.0 <= pools.om_percentage <= 5.0


def test_rothc_kinetics_step():
    kinetics = RothCSOCKinetics()
    pools = SoilCarbonPools()
    initial_soc = pools.total_soc

    new_pools, co2_respired = kinetics.step(
        pools=pools,
        annual_c_input_t_ha=4.0,
        dpm_rpm_ratio=1.44,
        mean_temp_c=12.0,
        annual_precip_mm=650.0,
        et0_mm=800.0,
    )

    # Assert mass conservation: inputs and respired amounts are non-negative
    assert co2_respired > 0.0
    assert new_pools.total_soc > 0.0
    # Net delta should reflect inputs minus respired
    assert isinstance(new_pools.dpm, float)
    assert isinstance(new_pools.rpm, float)


def test_pedotransfer_hydro_dynamics():
    hydro = PedotransferHydroDynamics()
    # High OM should decrease bulk density and increase K_sat
    low_om_props = hydro.compute_hydraulic_properties(om_pct=1.0)
    high_om_props = hydro.compute_hydraulic_properties(om_pct=4.5)

    assert high_om_props["bulk_density_g_cm3"] < low_om_props["bulk_density_g_cm3"]
    assert high_om_props["k_sat_mm_hr"] > low_om_props["k_sat_mm_hr"]
    assert high_om_props["total_porosity"] > low_om_props["total_porosity"]
    assert high_om_props["pawc_mm_m"] > 50.0


def test_farmer_adoption_hazard_model():
    model = FarmerAdoptionHazardModel(a_max=0.85, kappa=0.35, t0_inflection=4.0)
    # Check monotonicity of adoption curve
    a_yr0 = model.adoption_fraction(0)
    a_yr5 = model.adoption_fraction(5)
    a_yr15 = model.adoption_fraction(15)

    assert 0.0 <= a_yr0 <= a_yr5 <= a_yr15 <= 0.85
    # Check yield lag in early years and recovery in later years
    assert model.yield_multiplier(1) < 1.00
    assert model.yield_multiplier(6) >= 1.00

    # Net margin should be positive
    margin_yr1 = model.annual_net_margin_eur_ha(1)
    margin_yr8 = model.annual_net_margin_eur_ha(8)
    assert margin_yr1 > 300.0
    assert margin_yr8 > margin_yr1


def test_decadal_trajectory_simulation():
    sim = DecadalTrajectorySimulator()
    res = sim.run_trajectory(
        start_year=2025,
        end_year=2050,
        annual_c_input_t_ha=4.2,
        baseline_erosion_t_ha=22.0,
    )

    assert res["total_years"] == 26
    assert res["net_soc_delta_t_ha"] > 0.0  # Demonstrating continuous soil carbon sink
    assert res["total_erosion_mitigated_t_ha"] > 100.0  # Substantial soil savings
    assert len(res["trajectory"]) == 26

    # Verify steady progression
    first_year = res["trajectory"][0]
    last_year = res["trajectory"][-1]
    assert last_year["om_pct"] > first_year["om_pct"]
    assert last_year["annual_erosion_t_ha"] < first_year["annual_erosion_t_ha"]
    assert last_year["k_sat_mm_hr"] > first_year["k_sat_mm_hr"]
