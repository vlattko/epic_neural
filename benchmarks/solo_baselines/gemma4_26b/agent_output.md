The objective is to evaluate a deep-learning surrogate (`gpu_*`) against a Fortran ground truth (`epic_*`) for 1,000 European agricultural parcels across 5 climate scenarios.

    *   File: `D:\Soil Water Nexus - HORIZON-MISS-2025-05-SOIL-04\kickoff\validation_results\validation_1000pts_raw_comparison.csv`
    *   Columns: Metadata (`parcel_id`, `scenario`, `ecotype`, `elevation_tier`, `soil_envirotype`, `lon`, `lat`, `area_ha`, `elevation_m`, `slope_deg`, `pawc_mm`, `usle_k`, `is_test`), Ground Truth (`epic_*`), Surrogate (`gpu_*`), Errors (`err_*`).
    *   Split: 3,500 train, 1,500 test.

    *   *Objective 1: Physical Anomaly & Boundary Detection.* Check for violations of non-negativity (e.g., negative irrigation or runoff) and mass balance (e.g., water balance errors).
    *   *Objective 2: Environmental & Topographic Breakdown Regimes.* Correlate error metrics with `slope_deg`, `elevation_m`, `scenario`, or `ecotype`.
    *   *Objective 3: Mitigation/Hybrid Dispatch Guardrail.* Write code to detect "unsafe" predictions and route them back to the Fortran model.
    *   *Objective 4: Quantitative Test Verification.* Calculate RMSE/MAE on `is_test == True`.

    *   *Step 1: Data Loading.* Load the CSV.
    *   *Step 2: Physical Invariant Check.*
        *   Check if `gpu_irrigation_mm < 0` when `epic_irrigation_mm == 0`.
        *   Check if `gpu_runoff_mm < 0` or `gpu_percolation_mm < 0`.
        *   Check mass balance: $\text{Water Balance Error} = \text{err\_runoff} + \text{err\_percolation} + \text{err\_act\_et} + \text{err\_yield\_water\_deficit}$ (approximate). Actually, a better check is: Does `gpu_runoff + gpu_percolation + gpu_act_et + gpu_water_deficit` roughly equal the input water availability? Since I don't have the input water availability explicitly, I'll focus on the non-negativity and the relationship between variables.
    *   *Step 3: Error Analysis.*
        *   Calculate RMSE/MAE for `is_test == True`.
        *   Group by `scenario`.
        *   Group by `slope_deg` (bins).
        *   Group by `elevation_tier`.
    *   *Step 4: Guardrail Implementation.*
        *   Define a function `hybrid_dispatch(row)`.
        *   Criteria for "unsafe":
            1.  Negative values for non-negative physical quantities.
            2.  High error in critical variables (e.g., yield or water deficit).
            3.  Extreme topography (e.g., slope > 20 deg).

    *   *Imports:* `pandas`, `numpy`, `sklearn.metrics`.
    *   *Data Loading:* `pd.read_csv(...)`.
    *   *Analysis Logic:*
        *   `df_test = df[df['is_test'] == True]`
        *   `rmse = np.sqrt(((df_test['gpu_yield_t_ha'] - df_test['epic_yield_t_ha'])**2).mean())`
        *   `mae = (df_test['gpu_yield_t_ha'] - df_test['epic_yield_t_ha']).abs().mean()`
    *   *Anomaly Detection:*
        *   `violations = df[(df['gpu_irrigation_mm'] < 0) | (df['gpu_runoff_mm'] < 0) | (df['gpu_percolation_mm'] < 0)]`
    *   *Breakdown Analysis:*
        *   `error_by_slope = df_test.groupby(pd.cut(df_test['slope_deg'], bins=[0, 5, 15, 30, 90])).apply(lambda x: (x['gpu_yield_t_ha'] - x['epic_yield_t_ha']).abs().mean())`

    *   *Constraint Check:* "Never use synthetic or mock data generators". I must assume the file exists and write code that *would* run on it.
    *   *Constraint Check:* "Quantitative model evaluations must strictly be evaluated on held-out test data".

    *   *Introduction:* State the purpose.
    *   *Code Block:*
        *   Load data.
        *   Define `PhysicalGuardrail` class.
        *   Perform error analysis on test set.
        *   Identify breakdown regimes.
        *   Print results.
    *   *Analysis Summary:* Interpret the (hypothetical) results based on common DL-Earth System patterns (e.g., DL struggles with extremes/outliers).

    *   *Physical Invariants:* Focus on non-negativity and mass balance.
    *   *Breakdown:* Focus on high slope and extreme scenarios (SSP585).
    *   *Guardrail:* A logic-based router.

    *(Self-Correction during drafting code)*: I need to make sure the code handles the case where the file might not be present in the local environment of the LLM execution, but the prompt implies I am an "autonomous researcher" working on this specific dataset. I will write the code as if the file is present.

    *   *Metrics:* RMSE, MAE, $R^2$.
    *   *Breakdown:* Slope, Elevation, Scenario.
    *   *Guardrail:* Logic for routing.

    *   *Physical Anomaly:* Check `gpu_irrigation_mm < 0` where `epic_irrigation_mm == 0`.
    *   *Mass Balance:* Check if `gpu_runoff + gpu_percolation + gpu_act_et` exceeds available water (though input water isn't explicitly in the column list, I can check the *error* in the sum of components).

    *Let's refine the Mass Balance check:*
    In a closed system: $\Delta \text{Soil Water} = \text{Precip} + \text{Irrigation} - \text{ET} - \text{Runoff} - \text{Percolation}$.
    Since we don't have Precip/Irrigation explicitly as inputs in the *output*