"""Which soil/climate inputs actually drive EPIC yield and USLE erosion?

Fits :class:`RandomForestRegressor` (erosion) and :class:`GradientBoostingRegressor`
(yield) trees on the Agro-Hydrological Soil-Water Nexus EPIC validation table and reads the resulting
feature importances as "driver sensitivity": how strongly each physical soil
property (``slope_deg``, ``elevation_m``, ``pawc_mm``, ``sand``, ``silt``,
``clay``, ``bd``, ``soc``, ``ph``, ``usle_k``) and each climate stress
(``epic_water_stress_days``, ``epic_temp_stress_days``, ``epic_water_deficit_mm``)
moves the output.

Public API:

* :func:`analyze_erosion_drivers` - driver importances for
  ``epic_usle_erosion_t_ha`` (RandomForest primary importances).
* :func:`analyze_yield_drivers` - driver importances for
  ``epic_yield_t_ha`` (GradientBoosting primary importances).
* :func:`compute_scenario_divergence` - per-driver sensitivity shift of each
  future scenario (``ssp126_conservation``, ``ssp245_conventional``,
  ``ssp585_conventional``) against ``historical_baseline``.

Run from a terminal::

    python src/analytics/drivers.py                 # all scenarios, both targets
    python src/analytics/drivers.py --scenario ssp585_conventional --target erosion
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# --- reuse the loader from src/data/soil_loader ------------------------------------
# Preferred: repo root on sys.path so ``src`` is a normal package. The fallback
# reads the loader straight off disk so this module also works when it is loaded
# by file path (as tests/test_drivers.py does).
try:
    from src.data.soil_loader import load_soil_data
except ImportError:  # pragma: no cover - standalone / path-loaded use
    _loader_path = Path(__file__).resolve().parents[1] / "data" / "soil_loader.py"
    if not _loader_path.exists():  # pragma: no cover
        raise
    _spec = importlib.util.spec_from_file_location("soil_loader", _loader_path)
    _loader = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_loader)
    load_soil_data = _loader.load_soil_data

# --- the drivers we attribute the outputs to ---------------------------------------
PHYSICAL_FEATURES = (
    "slope_deg",
    "elevation_m",
    "pawc_mm",
    "sand",
    "silt",
    "clay",
    "bd",
    "soc",
    "ph",
    "usle_k",
)
CLIMATE_FEATURES = (
    "epic_water_stress_days",
    "epic_temp_stress_days",
    "epic_water_deficit_mm",
)
FEATURES = PHYSICAL_FEATURES + CLIMATE_FEATURES

FEATURE_CLUSTER = {feature: "soil_physical" for feature in PHYSICAL_FEATURES}
FEATURE_CLUSTER.update({feature: "climate_stress" for feature in CLIMATE_FEATURES})

EROSION_TARGET = "epic_usle_erosion_t_ha"
YIELD_TARGET = "epic_yield_t_ha"
TARGET_COLUMNS = {"erosion": EROSION_TARGET, "yield": YIELD_TARGET}

BASELINE_SCENARIO = "historical_baseline"
FUTURE_SCENARIOS = ("ssp126_conservation", "ssp245_conventional", "ssp585_conventional")

MIN_TRAIN_ROWS = 4
IMPORTANCE_EPS = 1e-12

RANDOM_STATE = 42
RF_PARAMS = {
    "n_estimators": 150,
    "max_features": 0.6,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}
GB_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 1.0,
    "random_state": RANDOM_STATE,
}

DRIVER_COLUMNS = (
    "feature",
    "cluster",
    "target",
    "primary_model",
    "importance",
    "rf_importance",
    "gb_importance",
    "rank",
    "n_rows",
    "n_parcels",
)
DIVERGENCE_COLUMNS = (
    "feature",
    "cluster",
    "target",
    "scenario",
    "baseline_importance",
    "scenario_importance",
    "delta_importance",
    "sign_of_shift",
    "baseline_rank",
    "scenario_rank",
    "rank_shift",
    "n_rows",
    "n_parcels",
    "baseline_n_rows",
    "baseline_n_parcels",
)

__all__ = [
    "FEATURES",
    "PHYSICAL_FEATURES",
    "CLIMATE_FEATURES",
    "EROSION_TARGET",
    "YIELD_TARGET",
    "BASELINE_SCENARIO",
    "FUTURE_SCENARIOS",
    "DRIVER_COLUMNS",
    "DIVERGENCE_COLUMNS",
    "analyze_erosion_drivers",
    "analyze_yield_drivers",
    "compute_scenario_divergence",
]


def _scenario_subset(df: pd.DataFrame, scenario: str | None) -> pd.DataFrame:
    """Rows for one scenario (every row when ``scenario`` is None).

    Raises:
        ValueError: If ``scenario`` names a scenario absent from the frame.
    """
    if scenario is None:
        return df
    if "scenario" not in df.columns:
        raise ValueError(
            f"frame has no 'scenario' column, so scenario {scenario!r} cannot be selected."
        )

    values = df["scenario"].astype(str)
    subset = df.loc[values == str(scenario)]
    if subset.empty:
        available = ", ".join(sorted(set(values)))
        raise ValueError(
            f"scenario {scenario!r} is not in the frame; available scenarios: {available}"
        )
    return subset


def _fit_importances(df: pd.DataFrame, target_column: str, features=FEATURES):
    """Fit both forests and return ``(rf_importances, gb_importances, n_rows, n_parcels)``.

    Rows with a missing feature or missing target value are dropped before fitting,
    and the ``err_*`` comparison columns are never used as drivers.

    Raises:
        ValueError: If a requested column is absent, or fewer than
            :data:`MIN_TRAIN_ROWS` complete rows remain.
    """
    columns = [*features, target_column]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {', '.join(missing)}")

    parcel_column = "parcel_id" if "parcel_id" in df.columns else None
    work_columns = [*columns, parcel_column] if parcel_column else columns

    work = df.loc[:, work_columns].copy()
    for column in columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    complete = work.dropna(axis=0, how="any", subset=columns)

    n_rows = len(complete)
    if n_rows < MIN_TRAIN_ROWS:
        raise ValueError(
            f"need at least {MIN_TRAIN_ROWS} complete rows to fit the driver models, "
            f"found {n_rows}"
        )

    X = complete.loc[:, list(features)].to_numpy(dtype=float)
    y = complete[target_column].to_numpy(dtype=float)

    rf = RandomForestRegressor(**RF_PARAMS).fit(X, y)
    gb = GradientBoostingRegressor(**GB_PARAMS).fit(X, y)

    n_parcels = int(complete[parcel_column].nunique()) if parcel_column else n_rows
    return rf.feature_importances_, gb.feature_importances_, n_rows, n_parcels


def _driver_table(
    df: pd.DataFrame,
    *,
    target_label: str,
    primary_model: str,
    scenario: str | None = None,
    target_column: str | None = None,
    features=FEATURES,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Shared tidy driver table behind the two public ``analyze_*`` functions."""
    target_column = TARGET_COLUMNS.get(target_label, target_column)
    if target_column is None:
        raise ValueError(f"unknown target {target_label!r}; choose from {sorted(TARGET_COLUMNS)}")

    subset = _scenario_subset(df, scenario)
    rf_importances, gb_importances, n_rows, n_parcels = _fit_importances(subset, target_column, features)

    primary = {"rf": rf_importances, "gb": gb_importances}
    if primary_model not in primary:
        raise ValueError(f"unknown primary_model {primary_model!r}; choose from 'rf' or 'gb'")

    table = pd.DataFrame(
        {
            "feature": list(features),
            "cluster": [FEATURE_CLUSTER[feature] for feature in features],
            "target": target_label,
            "primary_model": primary_model,
            "rf_importance": rf_importances,
            "gb_importance": gb_importances,
            "importance": primary[primary_model],
            "n_rows": n_rows,
            "n_parcels": n_parcels,
        }
    )
    table = table.sort_values("importance", ascending=False, kind="mergesort")
    table["rank"] = np.arange(1, len(table) + 1)

    if top_n is not None and top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    return table[list(DRIVER_COLUMNS)].reset_index(drop=True).head(top_n)


def analyze_erosion_drivers(
    df: pd.DataFrame,
    scenario: str | None = None,
    *,
    target_column: str = EROSION_TARGET,
    top_n: int | None = None,
    features=FEATURES,
) -> pd.DataFrame:
    """Rank the drivers of ``epic_usle_erosion_t_ha`` with RandomForestRegressor.

    Args:
        df: Frame produced by :func:`load_soil_data` (or any frame carrying the
            target plus the driver columns in :data:`FEATURES`).
        scenario: Restrict the fit to one scenario (``historical_baseline``, an
            ``ssp*`` scenario, ...). ``None`` fits every row at once.
        target_column: Override the erosion target column if needed.
        top_n: Keep only the ``top_n`` strongest drivers (all drivers by default).
        features: Driver columns to attribute (defaults to :data:`FEATURES`).

    Returns:
        One row per driver column with ``DRIVER_COLUMNS``: ``feature``, ``cluster``
        (``soil_physical`` / ``climate_stress``), ``target`` (``"erosion"``),
        ``primary_model`` (``"rf"``), ``importance`` (= ``rf_importance``), the
        secondary ``gb_importance``, ``rank`` (1 = strongest driver) and the fit
        sizes ``n_rows`` / ``n_parcels``. Sorted by ``importance`` descending.

    Raises:
        ValueError: If a required column is missing, the scenario is absent, fewer
            than :data:`MIN_TRAIN_ROWS` complete rows remain, or ``top_n < 1``.
    """
    return _driver_table(
        df,
        target_label="erosion",
        primary_model="rf",
        scenario=scenario,
        target_column=target_column,
        features=features,
        top_n=top_n,
    )


def analyze_yield_drivers(
    df: pd.DataFrame,
    scenario: str | None = None,
    *,
    target_column: str = YIELD_TARGET,
    top_n: int | None = None,
    features=FEATURES,
) -> pd.DataFrame:
    """Rank the drivers of ``epic_yield_t_ha`` with GradientBoostingRegressor.

    Args:
        df: Frame produced by :func:`load_soil_data` (or any compatible frame).
        scenario: Restrict the fit to one scenario; ``None`` fits every row.
        target_column: Override the yield target column if needed.
        top_n: Keep only the ``top_n`` strongest drivers (all drivers by default).
        features: Driver columns to attribute (defaults to :data:`FEATURES`).

    Returns:
        One row per driver column with ``DRIVER_COLUMNS``, ``importance`` taken
        from GradientBoostingRegressor (``primary_model == "gb"``) and the
        RandomForest importances kept in ``rf_importance`` for cross-checking.
        Sorted by ``importance`` descending.

    Raises:
        ValueError: As for :func:`analyze_erosion_drivers`.
    """
    return _driver_table(
        df,
        target_label="yield",
        primary_model="gb",
        scenario=scenario,
        target_column=target_column,
        features=features,
        top_n=top_n,
    )


def compute_scenario_divergence(
    df: pd.DataFrame,
    *,
    baseline: str = BASELINE_SCENARIO,
    scenarios=FUTURE_SCENARIOS,
    targets=("erosion", "yield"),
    features=FEATURES,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Driver sensitivity shifts of future scenarios against the baseline.

    Each future scenario is fitted separately against ``baseline``
    (``historical_baseline`` by default) for each target, and the per-driver
    importances are subtracted: a positive ``delta_importance`` means the driver
    matters *more* in that scenario than it did in the baseline, and a negative
    ``rank_shift`` means it moved up the driver ranking.

    Args:
        df: Frame produced by :func:`load_soil_data`.
        baseline: Reference scenario (defaults to :data:`BASELINE_SCENARIO`).
        scenarios: Comparison scenarios (defaults to :data:`FUTURE_SCENARIOS`).
        targets: ``"erosion"`` and/or ``"yield"`` (both by default).
        features: Driver columns to attribute (defaults to :data:`FEATURES`).
        top_n: Keep only the ``top_n`` largest absolute shifts per
            scenario/target pair (all drivers by default).

    Returns:
        Tidy long frame with one row per ``scenario``/``target``/``feature``
        combination and ``DIVERGENCE_COLUMNS``: the identity fields ``feature``,
        ``cluster``, ``target``, ``scenario``, the compared importances
        ``baseline_importance`` / ``scenario_importance``, their difference
        ``delta_importance``, ``sign_of_shift`` (``+1`` stronger, ``-1`` weaker,
        ``0`` unchanged), the ranks in each fit (``baseline_rank``,
        ``scenario_rank``, ``rank_shift = baseline_rank - scenario_rank``) and
        the fit sizes ``n_rows`` / ``n_parcels`` / ``baseline_n_rows`` /
        ``baseline_n_parcels``. Sorted by absolute ``delta_importance``
        descending.

    Raises:
        ValueError: If a required column is missing, a named scenario is absent,
            an unknown target is requested, fewer than :data:`MIN_TRAIN_ROWS`
            complete rows remain for a fit, or ``top_n < 1``.
    """
    if top_n is not None and top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    unknown_targets = [label for label in targets if label not in TARGET_COLUMNS]
    if unknown_targets:
        raise ValueError(f"unknown targets: {', '.join(unknown_targets)}")
    if top_n is not None and top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    # Baseline importances + ranks per target.
    base_rows: dict[str, dict[str, tuple[float, int]]] = {}
    base_sizes: dict[str, tuple[int, int]] = {}
    for label in targets:
        table = _driver_table(
            df,
            target_label=label,
            primary_model="rf" if label == "erosion" else "gb",
            scenario=baseline,
            features=features,
        )
        base_rows[label] = {
            row["feature"]: (row["importance"], int(row["rank"])) for row in table.to_dict("records")
        }
        base_sizes[label] = (int(table["n_rows"].iloc[0]), int(table["n_parcels"].iloc[0]))

    records: list[dict] = []
    for label in targets:
        primary_model = "rf" if label == "erosion" else "gb"
        for scenario in scenarios:
            table = _driver_table(
                df,
                target_label=label,
                primary_model=primary_model,
                scenario=scenario,
                features=features,
            )
            pair_rows = []
            for row in table.to_dict("records"):
                baseline_importance, baseline_rank = base_rows[label][row["feature"]]
                delta = row["importance"] - baseline_importance
                pair_rows.append(
                    {
                        "feature": row["feature"],
                        "cluster": row["cluster"],
                        "target": label,
                        "scenario": scenario,
                        "baseline_importance": baseline_importance,
                        "scenario_importance": row["importance"],
                        "delta_importance": delta,
                        "sign_of_shift": int(np.sign(delta)) if abs(delta) > IMPORTANCE_EPS else 0,
                        "baseline_rank": baseline_rank,
                        "scenario_rank": int(row["rank"]),
                        "rank_shift": baseline_rank - int(row["rank"]),
                        "n_rows": row["n_rows"],
                        "n_parcels": row["n_parcels"],
                        "baseline_n_rows": base_sizes[label][0],
                        "baseline_n_parcels": base_sizes[label][1],
                    }
                )
            if top_n is not None:
                pair_rows.sort(key=lambda record: abs(record["delta_importance"]), reverse=True)
                pair_rows = pair_rows[:top_n]
            records.extend(pair_rows)

    divergence = pd.DataFrame.from_records(records)
    divergence = divergence.sort_values(
        "delta_importance", key=lambda column: column.abs(), ascending=False, kind="mergesort"
    )
    return divergence[list(DIVERGENCE_COLUMNS)].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the driver tables and the baseline divergence."""
    parser = argparse.ArgumentParser(description="Attribute EPIC yield and USLE erosion to soil/climate drivers")
    parser.add_argument("--csv", default=None, help="validation CSV to load (defaults to the loader's path)")
    parser.add_argument("--scenario", default=None, help="restrict the driver tables to one scenario")
    parser.add_argument("--target", choices=("erosion", "yield", "both"), default="both", help="target to analyse")
    parser.add_argument("--baseline", default=BASELINE_SCENARIO, help="divergence reference scenario")
    parser.add_argument("--scenarios", nargs="*", default=list(FUTURE_SCENARIOS), help="divergence comparison scenarios")
    parser.add_argument("--top", type=int, default=None, help="keep only the top N drivers")
    args = parser.parse_args(argv)

    df = load_soil_data(args.csv) if args.csv else load_soil_data()
    targets = ("erosion", "yield") if args.target == "both" else (args.target,)

    print(f"{len(df):,} rows loaded")
    for label in targets:
        table = (
            analyze_erosion_drivers(df, args.scenario, top_n=args.top)
            if label == "erosion"
            else analyze_yield_drivers(df, args.scenario, top_n=args.top)
        )
        print(f"\n[{label}] {table.to_string(float_format=lambda value: f'{value:.5f}')}")

    divergence = compute_scenario_divergence(df, baseline=args.baseline, scenarios=args.scenarios, top_n=args.top)
    print(f"\n[divergence] {divergence.to_string(float_format=lambda value: f'{value:.5f}')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
