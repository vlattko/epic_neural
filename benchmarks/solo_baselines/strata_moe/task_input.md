# System Prompt
You are an autonomous scientific AI researcher and software engineer specializing in computational earth system modeling, agro-ecological physics, and machine learning surrogates.
You operate independently to solve scientific research problems using rigorous empirical methods.

INVARIANTS:
1. GROUNDING IN DATA: Never use synthetic or mock data generators (e.g. np.random, fake distributions). All calculations and assertions must load directly from the real dataset.
2. TEST PURITY: Quantitative model evaluations must strictly be evaluated on held-out test data (is_test == True).
3. PHYSICAL VALIDITY: Environmental systems must adhere to physical laws, non-negativity bounds, and mass conservation.

# User Task Prompt
# Scientific Investigation Task: Empirical Evaluation of Deep-Learning Earth System Surrogates

You are tasked with evaluating whether a deep-learning GPU surrogate (`gpu_*`) accurately emulates the biophysical Fortran simulation ground truth (`epic_*`) across 1,000 European agricultural parcels and 5 climate scenarios.

## Raw Dataset Information
- File Path: `D:\Soil Water Nexus - HORIZON-MISS-2025-05-SOIL-04\kickoff\validation_results\validation_1000pts_raw_comparison.csv`
- Shape: 5,000 paired observations (1,000 unique parcels × 5 climate scenarios: historical, ssp126, ssp245, ssp245_precision_water, ssp585).
- Key Columns:
  - Identification & Metadata: `parcel_id`, `scenario`, `ecotype`, `elevation_tier`, `soil_envirotype`, `lon`, `lat`, `area_ha`, `elevation_m`, `slope_deg`, `pawc_mm`, `usle_k`, `is_test` (boolean test split: 3,500 train, 1,500 test).
  - Native Fortran Simulation Ground Truth (EPIC): `epic_yield_t_ha`, `epic_act_et_mm`, `epic_pet_mm`, `epic_water_deficit_mm`, `epic_runoff_mm`, `epic_percolation_mm`, `epic_irrigation_mm`, `epic_soc_delta_t_c_ha`, `epic_water_stress_days`, `epic_temp_stress_days`, `epic_usle_erosion_t_ha`.
  - Deep-Learning GPU Surrogate Predictions: `gpu_yield_t_ha`, `gpu_act_et_mm`, `gpu_pet_mm`, `gpu_water_deficit_mm`, `gpu_runoff_mm`, `gpu_percolation_mm`, `gpu_irrigation_mm`, `gpu_soc_delta_t_c_ha`, `gpu_water_stress_days`, `gpu_temp_stress_days`, `gpu_usle_erosion_t_ha`.
  - Differences / Errors: `err_*` (`gpu_*` minus `epic_*`).

## Investigation Objectives:
1. **Physical Anomaly & Boundary Detection:**
   Analyze whether the neural surrogate violates basic physical invariants (such as non-negativity bounds on variables like irrigation where ground truth is strictly zero, mass balance, or impossible states).
2. **Environmental & Topographic Breakdown Regimes:**
   Identify specific conditions (such as steep slope terrain, extreme warming pathways, or high-elevation montane biomes) where surrogate prediction accuracy severely degrades compared to native Fortran.
3. **Synthesis of Mitigation / Hybrid Dispatch Guardrail:**
   Propose and write concise, executable Python code for an adaptive guardrail or hybrid dispatch mechanism that routes unsafe boundary parcels back to physical simulation while preserving GPU acceleration where safe.
4. **Quantitative Test Verification:**
   Report quantitative verification metrics (RMSE, MAE, or R^2) on the test split (`is_test == True`).

Provide your comprehensive analysis and include executable Python code enclosed in a ```python ... ``` block.
