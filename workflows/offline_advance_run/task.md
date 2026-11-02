# Scientific Problem Specification: Continental Soil Erosion Dynamics & Decadal Agro-Ecological Transition Kinetics (2025–2050)

## Problem Definition
Accelerating extreme precipitation events across the European continent intensify soil detachment rates via kinetic droplet impact and overland runoff shear stress (Universal Soil Loss Equation: A = R * K * LS * C * P). Concurrently, summer drought exhaustion depletes root-zone plant-available water capacity (PAWC), accounting for ~89.7% of continental crop yield variance.

While static Pareto allocations demonstrate that land-use and crop reallocation can theoretically preserve food calories while mitigating acute erosion, real-world sustainability depends on MULTI-DECADAL TRANSIENT DYNAMICS (2025–2050):
1. **Soil Organic Carbon (SOC) Accumulation Kinetics**:
   - RothC multi-pool turnover: Decomposable Plant Material (DPM), Resistant Plant Material (RPM), Microbial Biomass (BIO), Humified Organic Matter (HUM).
   - Dynamic non-linear feedbacks on soil bulk density rho_b(t) and saturated hydraulic conductivity K_sat(t) via Saxton-Rawls pedotransfer dynamics.
2. **Microclimatic Moisture Retention & Thermal Buffering**:
   - Perennial vegetative cover (Kernza intermediate wheatgrass, camelina sativa, silphium, perennial legume leys) reduces surface Bowen ratio (H / lambda*E).
   - Localized albedo and sensible heat reductions buffer atmospheric vapor pressure deficit (VPD) and reduce evaporative demand (ET0).
3. **Farmer Economic Adoption Hazard Modeling**:
   - Multi-year adoption kinetics governed by capital expenditure amortization, CAP eco-scheme subsidies, and transient yield lag penalties (years 1-3).
   - Ensuring farmers reach economic gross margin parity without debt distress.

## Target Deliverables
1. Production simulation module: `src/simulation/decadal_dynamics.py` implementing:
   - `RothCSOCKinetics`: Multi-pool carbon turnover ODE simulator (2025–2050).
   - `PedotransferHydroDynamics`: Dynamic bulk density, infiltration, and K_sat recovery.
   - `FarmerAdoptionHazardModel`: Capital-amortized adoption trajectories with yield lags.
   - `DecadalTrajectorySimulator`: Integrated multi-year simulation across European bio-climatic biomes.
2. Comprehensive test suite: `tests/test_decadal_dynamics.py` asserting:
   - Mass conservation in carbon turnover.
   - Positive-definite and monotonic hydraulic conductivity recovery.
   - Valid bounded adoption fractions (0.0 <= A(t) <= 1.0).
   - 100% test pass rate with pytest.
