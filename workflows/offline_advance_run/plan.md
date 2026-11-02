## 1. Architectural Objective

The target system is a deterministic, offline, multi-agent scientific simulation cluster for European continental soil-erosion and agro-ecological transition dynamics from **2025 to 2050**.

The simulator must couple:

1. **Soil organic carbon turnover** via a RothC-style multi-pool ODE system.
2. **Hydraulic recovery** via dynamic bulk density, infiltration, and saturated hydraulic conductivity.
3. **Farmer adoption behavior** via capital-amortized hazard dynamics with yield-lag penalties.
4. **Continental biome heterogeneity** via biome-specific climate, soil, crop, and economic parameters.
5. **Erosion and land-use feedbacks** via a dynamic Universal Soil Loss Equation formulation.

The architecture must be testable, mass-conserving, numerically stable, and physically bounded.

---

# 2. Top-Level Module Architecture

Target production module:

```text
src/simulation/decadal_dynamics.py
```

Target test module:

```text
tests/test_decadal_dynamics.py
```

Recommended internal organization:

```text
src/simulation/
    decadal_dynamics.py
    core/
        time.py
        units.py
        validation.py
        numerical.py
    soil/
        rothc.py
        pedotransfer.py
        erosion.py
    hydrology/
        moisture.py
        vpd.py
        et0.py
        bowen.py
    economics/
        adoption.py
        margins.py
        capital.py
    biomes/
        biome_profiles.py
        climate_drivers.py
    simulator/
        trajectory.py
        state.py
        outputs.py
```

The public production module may expose the four required classes:

```python
RothCSOCKinetics
PedotransferHydroDynamics
FarmerAdoptionHazardModel
DecadalTrajectorySimulator
```

---

# 3. Time Domain and Simulation Clock

## 3.1 Simulation Interval

```text
t ∈ [2025, 2050]
```

Recommended annual resolution:

```text
Δt = 1 year
```

Recommended internal numerical substepping:

```text
Δt_internal ∈ [0.01, 0.25] years
```

Use adaptive Runge-Kutta integration for ODE stability, with annual forcing updates.

## 3.2 Time State

```python
class SimulationClock:
    t0: float = 2025.0
    t1: float = 2050.0
    dt_output: float = 1.0
    dt_internal: float = 0.1
    max_substeps: int = 100
```

Validation:

```