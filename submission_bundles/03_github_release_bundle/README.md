# Autonomous Multi-Agent Scientific Discovery with Long-Context Gemma 4 and Decentralized Heterogeneous Computing: Unraveling the European Soil-Water Nexus

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-76%20passed%20(100%25)-brightgreen.svg)](tests/)
[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](LICENSE)
[![Context Window](https://img.shields.io/badge/Context-262k%20Tokens-purple.svg)](paper/)
[![Offline](https://img.shields.io/badge/Cloud%20Dependency-0%25%20(100%25%20Offline)-orange.svg)](paper/)

Official codebase and reproducible evaluation harness for the paper:  
**"Autonomous Multi-Agent Scientific Discovery with Long-Context Gemma 4 and Decentralized Heterogeneous Computing: Unraveling the European Soil-Water Nexus"**  
*Authored by Vlatko Galić (Agricultural Institute Osijek), Strata Lead Architect, Gemma 4 26B-A4B-it, Qwen 3.8 27B, and Strata Coder IQ1_M.*

---

## 1. System Overview: "Don't Sell Your Old PC"

Monolithic conversational AI agents routinely fail when deployed on high-dimensional earth system challenges because single-context loops suffer from conversational drift, hallucination of physical invariants, and prohibitive cloud inference quotas. 

This repository implements a **100% offline, decentralized multi-agent scientific discovery cluster** running across consumer hardware on a local gigabit router:
- **Primary Workstation (Dual RTX 5060 Ti Blackwell, 128 GB RAM):**
  - **Strata MoE (`Qwen3.8-Flash-Next-IQ3_XXS`, Port 8080):** Lead Scientific Architect & Chief Supervisor.
  - **Gemma 4 (`gemma-4-26B-A4B-it`, Port 8081, CUDA0):** Biophysical Modeling & Review Specialist (262k context).
  - **Qwen 27B (`Qwen3.8-27B`, Port 8082, CUDA1):** Agronomic Economics Ideator (262k context).
- **Secondary Router PC (RTX 4060 Ti 16 GB, 32 GB RAM):**
  - **Strata Coder (`qwen3.8-flash-next-coder-iq1_m`, 192.168.1.120:8080):** Autonomous Coding & Repair Worker.
- **Inter-Agent Coordination:** Stateless, append-only disk ledgers adhering to the **Memdoor discipline** (`task.md`, `plan.md`, `notes.md`, `review.md`).

---

## 2. Key Scientific Breakthroughs

### The Three Physical Failure Regimes of Neural Surrogates
Evaluating 5,000 paired EPIC Fortran vs deep-learning surrogate runs across continental agricultural parcels unmasked 3 critical physical breakdown regimes:
1. **Zero-Inflation Non-Negativity Violation (Dryland Irrigation):** Neural surrogates output negative irrigation in 1,614 out of 4,000 dryland runs (min -4.54 mm), hallucinating phantom water withdrawals ($R^2 = 0.0000$).
2. **Alpine Extremes Under High Warming (SSP5-8.5):** In high-elevation Alpine soils, SOC delta accuracy collapses to $R^2 = 0.1659$ and water stress days to $R^2 = 0.3075$.
3. **Topographic Runoff Shear Amplification:** On rolling slopes ($8^\circ\text{–}15^\circ$), USLE erosion prediction error balloons to $37.35\text{ t/ha/yr}$.

### The Physics-Informed Adaptive Hybrid Dispatcher
The agent collective synthesized an adaptive routing guardrail:
- **93.1% Safe GPU Allocation (4,655 evaluations):** Executed at >4,000 parcels/sec on local GPUs.
- **6.9% Fortran Fallback (345 evaluations):** Routed boundary parcels ($\theta \ge 8^\circ$ or Alpine SSP5-8.5) back to physical Fortran simulation.
- **Verified Impact:** 68.95% reduction in erosion RMSE, 100% elimination of negative irrigation, and 13.9x speedup over pure Fortran.

---

## 3. Solo-Agent Ablation Benchmark

| Evaluation Dimension | Gemma 4 26B (Solo) | Qwen 27B (Solo) | Strata MoE (Solo) | Strata Coder (Solo) | Team Collective (This Work) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean-Slate Score** | **4 / 6** | **4 / 6** | **4 / 6** | **3 / 6** | **6 / 6 (100% Perfect)** |
| **Code Executability** | Failed (Truncation) | Failed (No Code) | Failed (Syntax Error) | Failed (Incomplete) | **Passed (76/76 Tests)** |

---

## 4. One-Click Reproduction Guide

```bash
# 1. Clone repository
git clone https://github.com/vlattko/soil-water-nexus-agents.git
cd soil-water-nexus-agents

# 2. Run one-click verification harness
python submission_bundles/01_kaggle_submission/kaggle_quick_verification.py

# 3. Run complete test harness
pytest tests/ -q
# Expected: 76 passed, 18 warnings in ~314s
```

---

## 5. Citation Block

```bibtex
@article{galic2026autonomous,
  title={Autonomous Multi-Agent Scientific Discovery with Long-Context Gemma 4 and Decentralized Heterogeneous Computing: Unraveling the European Soil-Water Nexus},
  author={Gali{\'c}, Vlatko and Strata Lead Architect and Gemma 4 26B-A4B-it and Qwen 3.8 27B and Strata Coder IQ1_M},
  journal={arXiv preprint arXiv:2611.XXXXX},
  year={2026}
}
```
