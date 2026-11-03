Below is a complete, executable investigation workflow for the real CSV file you specified. It:

1. loads the real validation data,
2. evaluates the GPU surrogate only on the held-out test split (`is_test == True`),
3. detects physical anomalies and boundary violations,
4. identifies environmental/topographic regimes where surrogate accuracy degrades,
5. proposes and implements an adaptive guardrail / hybrid dispatch mechanism,
6. reports quantitative test metrics: RMSE, MAE, and R².

Because I do not have direct access to your local file from this environment, the code is written to be run on your machine against the real dataset. It does **not** use synthetic data and it evaluates metrics strictly on `is_test == True`.

```python
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

CSV_PATH = r"data/validation_results/validation_1000pts_raw_comparison.csv"

TARGETS = [
    "yield_t_ha",
    "act_et_mm",
    "pet_mm",
    "water_deficit_mm",
    "runoff_mm",
    "percolation_mm",
    "irrigation_mm",
    "soc_delta_t_c_ha",
    "water_stress_days",
    "temp_stress_days",
    "usle_erosion_t_ha",
]

NONNEGATIVE_TARGETS = [
    "yield_t_ha",
    "act_et_mm",
    "pet_mm",
    "water_deficit_mm",
    "runoff_mm",
    "percolation_mm",
    "irrigation_mm",
    "water_stress_days",
    "temp_stress_days",
    "usle_erosion_t_ha",
]

# Physical tolerance thresholds.
# These are intentionally conservative and can be tuned.
TOL_SMALL = 1e-6
TOL_MM = 0.5
TOL_DAYS = 0.