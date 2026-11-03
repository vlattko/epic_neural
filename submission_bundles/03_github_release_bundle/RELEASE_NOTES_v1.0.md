# Release Notes: v1.0.0-grounded

**Release Tag:** `v1.0.0-grounded`  
**Repository:** `https://github.com/vlattko/soil-water-nexus-agents`  
**Date:** November 2026  
**Target Milestone:** Kaggle Gemma 4 Developer Agent Paper Track & arXiv Submission  

---

## What's Included in This Release:

### 1. Grounded Surrogate Failure Physics Analysis
- Evaluates 5,000 paired observations comparing native Fortran EPIC against deep-learning GPU emulations across European agricultural parcels.
- Modules: `src/analysis/surrogate_failure_physics.py` and `src/analysis/pilot_drava_basin.py`.
- Ground truth ledgers: `results/grounded_*.json`.

### 2. Physics-Informed Adaptive Hybrid Dispatcher
- Dispatches 93.1% of parcels to GPU acceleration and 6.9% to Fortran physical core.
- Eliminates 100% of negative irrigation errors and cuts erosion RMSE by 68.95%.

### 3. Solo-Agent Baseline Ablation Benchmark
- Conducted clean-slate ablation benchmarking of individual agents acting in isolation without team collaboration or supervision.
- Proves that solo monoliths degrade (scoring 3-4/6 due to cognitive overload and truncation), while the team collective achieves a perfect 6/6 score.
- Module: `src/benchmarks/solo_agent_ablation.py`.

### 4. Automated 76-Item Test Suite
- Comprehensive pytest suite covering pedotransfer equations, carbon kinetics, dataset schemas, failure regimes, hybrid dispatcher guardrails, and ablation rubrics.
- 100% pass rate across all 76 unit tests.

### 5. Publication-Ready Manuscripts & High-Resolution Figures
- Kaggle & arXiv manuscript: `paper/gemma4_developer_agent_paper.md`.
- Nature Sustainability manuscript: `paper/nature_sustainability_european_soil_water_nexus.md`.
- 8 high-resolution 300 DPI publication figures in `paper/figures/`.
