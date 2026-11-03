Below is a complete, executable empirical evaluation script for the CSV you referenced. It does **not** use synthetic data; it loads the real CSV and computes all diagnostics directly from the dataset.

Because I cannot execute the file here, the script prints the required quantitative test-split metrics and anomaly summaries when run on your CSV.

```python
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 1. LOAD REAL DATASET
# =============================================================================

CSV_PATH = Path(
    r"data/validation_results/validation_1000pts_raw_comparison.csv"
)

df = pd.read_csv(CSV_PATH)

# Ensure boolean test flag is usable
if "is_test" not in df.columns:
    raise ValueError("Missing required column: is_test")

df["is_test"] = df["is_test"].astype(bool)

# =============================================================================
# 2. DEFINE VARIABLES AND PHYSICAL BOUNDS
# =============================================================================

# Variables that should be physically non-negative in this agro-hydrological system
NONNEG_VARS = [
    "epic_yield_t_ha",
    "epic_act_et_mm",
    "epic_pet_mm",
    "epic_water_deficit_mm",
    "epic_runoff_mm",
    "epic_percolation_mm",
    "epic_irrigation_mm",
    "epic_water_stress_days",
    "epic_temp_stress_days",
    "epic_usle_erosion_t_ha",
    "gpu_yield_t_ha",
    "gpu_act_et_mm",
    "gpu_pet_mm",
    "gpu_water_deficit_mm",
    "gpu_runoff_mm",
    "gpu_percolation_mm",
    "gpu_irrigation_mm",
    "gpu_water_stress_days",
    "gpu_temp_stress_days",
    "gpu_usle_erosion_t_ha",
]

# Variables where ground truth is strictly zero and surrogate should not invent positive values
ZERO_GROUND_TRUTH_VARS = [
    "epic_irrigation_mm",
    "epic_runoff_mm",
    "epic