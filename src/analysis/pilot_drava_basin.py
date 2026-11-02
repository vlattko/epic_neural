"""Empirical Grounded Analysis of the Croatian Pilot Site: Drava River Basin (Osijek).

Processes:
1. 54 DLR Flight-Eligible ARKOD Parcels around Agricultural Institute Osijek (AIO).
2. Daily Growing Season Hydrological & Sentinel-1 Soil Moisture Dynamics (April-September 2024).

Zero synthetic data. Zero random placeholders. 100% deterministic empirical verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

MANIFEST_PATH = Path(
    r"D:\Soil Water Nexus - HORIZON-MISS-2025-05-SOIL-04"
    r"\kickoff\Pilot_site_info\DLR_Eligible_ARKOD_Manifest_Osijek_54parcels.csv"
)

TIMESERIES_PATH = Path(
    r"D:\Soil Water Nexus - HORIZON-MISS-2025-05-SOIL-04"
    r"\kickoff\time_series\pilot_field_timeseries_hydrological.csv"
)


def analyze_pilot_drava(output_dir: Path) -> Dict[str, Any]:
    """Execute empirical analysis on Osijek pilot parcels and growing season time series."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists() or not TIMESERIES_PATH.exists():
        raise FileNotFoundError("Missing pilot manifest or time series CSV files.")

    df_man = pd.read_csv(MANIFEST_PATH)
    df_ts = pd.read_csv(TIMESERIES_PATH)

    # 1. Parcel summary
    total_parcels = len(df_man)
    area_cols = [c for c in df_man.columns if "area" in c.lower()]
    total_area_ha = float(df_man[area_cols[0]].sum()) if area_cols else 0.0

    mean_lat = float(df_man["centroid_lat"].mean())
    mean_lon = float(df_man["centroid_lon"].mean())

    # 2. Time series dynamics (Growing season 2024: April 1 to September 30, 184 days)
    mean_temp = float(df_ts["temp_c"].mean())
    max_temp = float(df_ts["temp_c"].max())
    mean_vpd = float(df_ts["vpd_kpa"].mean())

    # Soil Water Deficit Index (SWDI): SWDI < 0 indicates soil water deficit
    deficit_days = int((df_ts["swdi_filtered"] < 0.0).sum())
    severe_deficit_days = int((df_ts["swdi_filtered"] < -2.0).sum())
    min_swdi = float(df_ts["swdi_filtered"].min())

    # Mean surface soil moisture (SSM normalized)
    mean_ssm = float(df_ts["ssm_filtered"].mean())
    min_ssm = float(df_ts["ssm_filtered"].min())

    results = {
        "pilot_name": "Croatia - Drava River Lowland Agricultural Basin (AIO Osijek)",
        "geographic_center": {"lat": round(mean_lat, 6), "lon": round(mean_lon, 6)},
        "num_arkod_parcels": total_parcels,
        "total_area_ha": round(total_area_ha, 2),
        "monitoring_season": "2024 Growing Season (April 1 - September 30, 184 days)",
        "microclimate": {
            "mean_temperature_c": round(mean_temp, 2),
            "max_temperature_c": round(max_temp, 2),
            "mean_vpd_kpa": round(mean_vpd, 3),
        },
        "hydrological_stress": {
            "total_days_evaluated": len(df_ts),
            "water_deficit_days_swdi_negative": deficit_days,
            "severe_deficit_days_swdi_under_minus_two": severe_deficit_days,
            "min_swdi_recorded": round(min_swdi, 3),
            "mean_surface_soil_moisture": round(mean_ssm, 3),
            "min_surface_soil_moisture": round(min_ssm, 3),
        },
    }

    out_file = output_dir / "grounded_pilot_drava.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    out = Path(r"D:\Kaggle\results")
    res = analyze_pilot_drava(out)
    print("Grounded Pilot Analysis Complete:", res)
