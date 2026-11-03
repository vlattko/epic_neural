# Kaggle Submission Portal Guide and Metadata

**Target Competition:** Kaggle Gemma 4 Developer Agent Paper Track  
**Competition URL:** `https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview`  
**Submission Format:** Paper / Technical Report with Public Codebase & Verification Artifacts  

---

## Ready-to-Copy Form Fields for Kaggle Submission

### 1. Title
```text
Autonomous Multi-Agent Scientific Discovery with Long-Context Gemma 4 and Decentralized Heterogeneous Computing: Unraveling the European Soil-Water Nexus
```

### 2. Subtitle / Short Description
```text
A 100% offline, zero-cloud multi-agent collective on local consumer GPUs discovering the physical failure boundaries of deep-learning earth system surrogates and synthesizing an adaptive hybrid dispatcher.
```

### 3. Authors and Affiliations
- **Vlatko Galić** (Principal Investigator & Lead Agronomist, Agricultural Institute Osijek)
- **Strata Lead Architect** (Local Distributed Inference Cluster, Dual NVIDIA RTX 5060 Ti Blackwell Workstation)
- **Gemma 4 26B-A4B-it** (Biophysical Modeling & Review Specialist, CUDA0)
- **Qwen 3.8 27B** (Agronomic Economics Ideator, CUDA1)
- **Strata Coder IQ1_M** (Autonomous Coding Worker, Secondary Router PC, NVIDIA RTX 4060 Ti 16 GB)

### 4. Repository & Weight Links
- **GitHub Repository:** `https://github.com/vlatkogalic/soil-water-nexus-agents`
- **Gemma 4 Model Weights:** `https://huggingface.co/google/gemma-4-26B-A4B-it`
- **Qwen 3.8 27B Weights:** `https://huggingface.co/Qwen/Qwen3.8-27B`
- **Dataset Provenance:** Horizon Europe HORIZON-MISS-2025-05-SOIL-04 (*HydroDataCube4EU*)

### 5. Abstract
```text
Monolithic large language models frequently fail when deployed on high-dimensional, multi-constraint scientific challenges because single-context agent loops suffer from progressive conversational drift, ungrounded hallucination of physical invariants, and prohibitive cloud inference quotas. In this paper, we introduce a completely offline, decentralized, heterogeneous multi-agent architecture that replaces cloud orchestration with local consumer hardware, anchored by the foundational ethos: "Don't Sell Your Old PC".

Our architecture couples a modern primary workstation featuring Dual NVIDIA GeForce RTX 5060 Ti GPUs (Blackwell architecture, 32 GB total VRAM, 128 GB host RAM) with an older secondary consumer PC (RTX 4060 Ti 16 GB, 32 GB RAM) connected across a local gigabit router. The primary workstation hosts the Strata Lead Architect (Qwen3.8-Flash-Next-IQ3_XXS, 75.8 GB MoE) at port 8080 as chief scientific supervisor, alongside Gemma 4 (google/gemma-4-26B-A4B-it) on CUDA0 (port 8081) as biophysical modeling and review specialist, and Qwen 27B (Qwen/Qwen3.8-27B) on CUDA1 (port 8082) as agronomic economics ideator. The secondary router PC runs a dedicated Strata Coder daemon (qwen3.8-flash-next-coder-iq1_m) at 172.20.20.120:8080 for high-throughput code synthesis and automated repair loops. Every model operates at a massive 262,144 token context window, and all agents communicate strictly through stateless, append-only disk ledgers adhering to the Memdoor discipline.

We deploy this offline collective against an urgent problem in computational earth system science: discovering the physical failure boundaries of deep-learning surrogates and synthesizing an adaptive physics-informed hybrid dispatcher. Drawing upon 5,000 paired simulation runs (1,000 real European agricultural parcels evaluated across 5 IPCC climate pathways) from Horizon Europe HORIZON-MISS-2025-05-SOIL-04 (HydroDataCube4EU), the agent team unmasked three systemic physical failure regimes in deep neural surrogates: (1) Zero-inflation non-negativity violations in dryland irrigation (1,614 negative predictions down to -4.54 mm; phantom withdrawals in 97.3% of dryland cases), (2) Alpine and Montane extreme warming breakdown under SSP5-8.5 (SOC R2 collapses to 0.1659), and (3) Topographic runoff shear detachment amplification on rolling slopes (USLE erosion errors up to 37.35 t/ha/yr).

To resolve these failures, the agent collective synthesized an Adaptive Physics-Informed Guardrail and Hybrid Dispatcher: retaining 93.1% of parcels (4,655 evaluations) in the fast GPU Surrogate Safe Operating Space (SSOS), while automatically routing the 6.9% high-risk parcels (345 evaluations) back to the native Fortran EPIC simulation core. This hybrid architecture eliminated 100% of negative irrigation events, reduced USLE erosion RMSE by 68.95% (from 2.12 to 0.66 t/ha), slashed maximum erosion error from 37.35 to 5.31 t/ha, and cut SOC error by 17.41%. The findings were further validated against in-situ empirical observations from the Croatian pilot site in the Drava River Basin (Osijek: 54 ARKOD parcels across 449.71 ha, revealing that 144 out of 184 growing-season days suffered active root-zone soil moisture deficit).

To prove that the multi-agent collective provides an essential advantage over isolated models, we executed a rigorous solo-agent ablation benchmark under clean-slate isolation and zero data leakage. While solo monoliths scored only 3 to 4 out of 6 due to cognitive overload, chain-of-thought token exhaustion, and unclosed code truncations, the decentralized collective achieved a perfect 6 out of 6 score. The entire pipeline was developed, debugged, and verified 100% offline with zero synthetic placeholders, validated by a 76 of 76 automated test pass rate (100%).
```

### 6. Key Scientific & Engineering Highlights for Judges
- **100% Offline & Zero-Cloud:** Runs completely on local consumer hardware (Dual RTX 5060 Ti + recycled RTX 4060 Ti router PC).
- **262k Long Context:** Every agent operates with 262,144 tokens, enabling ingestion of complete empirical pedotransfer tables and physical models.
- **Strict Anti-Hallucination Invariant:** Zero mock data, zero synthetic random generators; deterministic verification via `FactCheckAuditEngine`.
- **Verified Scientific Breakthrough:** Discovered 3 empirical failure regimes in deep learning earth system surrogates across 5,000 paired EPIC Fortran runs; synthesized an adaptive hybrid dispatcher cutting erosion RMSE by 68.95% while keeping 93.1% on GPU acceleration.
- **Solo Ablation Benchmark:** Proven superiority over isolated solo agents (6/6 perfect score vs 3-4/6 for solo monoliths).
- **Automated Verification:** 76 of 76 automated unit tests passing (100% pass rate).

---

## Submission Checklist for Kaggle

1. [ ] Log in to Kaggle and navigate to the **Gemma 4 Developer Agent Paper Track** overview page.
2. [ ] Click **"Submit Paper"** / **"New Submission"**.
3. [ ] Upload [`paper_submission.md`](paper_submission.md) (or compile to PDF via Pandoc/Typst and upload `paper_submission.pdf`).
4. [ ] In the submission text box / description, paste the Title, Subtitle, and Abstract above.
5. [ ] Provide the GitHub repository URL (`https://github.com/vlatkogalic/soil-water-nexus-agents`).
6. [ ] Attach the 3 key grounded figures from `figures/`:
   - `fig1_grounded_failure_regimes.png`
   - `fig2_grounded_hybrid_guardrail.png`
   - `fig3_grounded_pilot_drava.png`
7. [ ] Include verification note: *"Judges can verify all findings in 1 command by running `python kaggle_quick_verification.py`"*.
