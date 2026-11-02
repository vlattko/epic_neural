## Iteration Round 1 Audit Notes
Timestamp: 2026-11-02 21:51:00

### Biophysical Auditor Evaluation (Gemma 4)
*   Role: ADVERSARIAL BIOPHYSICAL AUDITOR and Earth System Physicist (Gemma 4).
*   Task: Examine specific empirical findings (provided in the prompt) and provide a rigorous biophysical evaluation and explanation of the physical mechanisms driving three failure regimes.
*   Data Points:
    1.  Dryland irrigation: Negative irrigation (-4.54 mm), phantom irrigation (2,279 parcels), $R^2 = 0.0$.
    2.  Alpine/Montane (SSP5-8.5): SOC $\Delta R^2 = 0.1659$, Water Stress Days $R^2 = 0.3075$.
    3.  Steep Slopes (8-15 deg): Max USLE error 37.35 t/ha, MAE = 5.24 t/ha.
    4.  Hybrid Guardrail: 6.9% Fortran EPIC / 93.1% GPU, cuts erosion RMSE by 68.95%, eliminates negative irrigation.

    *   *Regime 1: Mass Balance/Non-Negativity Violations (Dryland Irrigation).*
        *   *The Symptom:* Negative irrigation values and phantom withdrawals.
        *   *The Physics:* Irrigation is a mass flux ($I$). In a mass balance equation ($\Delta S = P + I - ET - R$), $I$ must be $\ge 0$.
        *   *The Failure Mechanism:* Deep learning (DL) models are function approximators, not mass-conserving integrators. They minimize MSE/MAE, not physical constraints. In dryland regions where $P$ (precipitation) is low and $ET$ (evapotranspiration) is high, the model likely over-predicts $ET$ or under-predicts $P$ to match a target soil moisture state. To compensate for a "deficit" that doesn't exist in the training data, the optimizer pushes the $I$ parameter into the negative domain to satisfy the residual. This is a "mathematical compensation for physical deficit."

    *   *Regime 2: Extrapolation/Domain Shift (Alpine/Montane under SSP5-8.5).*
        *   *The Symptom:* $R^2$ collapse for SOC (Soil Organic Carbon) and Water Stress Days.
        *   *The Physics:* Alpine/Montane systems are governed by non-linear temperature-sensitive kinetics (Arrhenius-type) and snowmelt-driven hydrology. SSP5-8.5 represents an extreme warming scenario.
        *   *The Failure Mechanism:* "Out-of-distribution" (OOD) extrapolation. The training data likely lacks the extreme thermal/hydrological forcing of SSP5-8.5. The model fails to capture the non-linear feedback between temperature, permafrost degradation, and SOC mineralization rates. For water stress, the model likely fails to account for the shift from snow-dominated to rain-dominated regimes, leading to a breakdown in the temporal distribution of water availability. The $R^2$ collapse indicates the model is essentially predicting the mean, losing all predictive power for the variance.

    *   *Regime 3: Topographic Shear/Non-linear Gradient Errors (Steep Slopes).*
        *   *The Symptom:* Massive USLE (Universal Soil Loss Equation) error on 8-15 deg slopes.
        *   *The Physics:* Erosion is highly non-linear with respect to slope ($\theta$). USLE uses a factor $K$ and $LS$ (slope length/steepness). The relationship is often power-law or exponential.
        *   *The Failure Mechanism:* "Gradient-induced hallucination." Standard CNNs or MLPs struggle with high-frequency spatial gradients and non-linear scaling. If the model is trained on flat/gentle terrain, it treats the slope as a linear perturbation. However, at 8-15 degrees, shear stress and runoff velocity increase non-linearly. The model underestimates the kinetic energy of runoff, leading to massive under-predictions of sediment transport, or conversely, over-predicting erosion due to lack of "roughness" constraints.

    *   *The Hybrid Guardrail (The Solution):*
        *   *The Mechanism:* This is a "Physics-Informed Hybrid Architecture." It uses the GPU for the "easy" (low-gradient, within-distribution) 93.1% of the domain and uses the "hard" (high-gradient, OOD, or non-physical) 6.9% for a deterministic, mass-conserving Fortran EPIC (Erosion Processes in Controlled environments) model.
        *   *The Result:* By enforcing the physics (mass balance and USLE equations) on the problematic 6.9%, it prevents the error from propagating through the system, effectively "clipping" the catastrophic errors.

    *   Tone: Professional, uncompromising, technical, "Adversarial Auditor."
    *   Structure:
        1.  Executive Summary of Failure.
        2.  Detailed Audit of Regime 1 (Mass Balance).
        3.  Detailed Audit of Regime 2 (Extrapolation/Climate Sensitivity).
        4.  Detailed Audit of Regime 3 (Topographic Non-linearity).
        5.  Evaluation of the Hybrid Guardrail.

    *   *Check:* Did I mention the specific numbers? Yes.
    *   *Check:* Did I explain the *why* (mechanisms)? Yes.
    *   *Check:* Is the tone right? Yes. (Use

### Quantitative Auditor Evaluation (Qwen 27B)
We need answer user's request. Need evaluate statistical validity and operational feasibility of grounded findings: 5000 paired observations (3500 train/1500 test across 1000 parcels and 5 scenarios). Hybrid Dispatcher 93.1% GPU surrogate execution with 6.9% Fortran fallback. Osijek pilot site 54 parcels, 449.7 ha, 144/184 days soil water deficit. Audit sample size, test-set purity, operational efficiency tradeoffs.

We need produce rigorous statistical methodologist response. Need likely discuss paired observations, sample size, independence, clustering, multiple scenarios, spatial autocorrelation, scenario dependence, train/test isolation, leakage, OOD, power, confidence intervals, surrogate vs Fortran fallback, operational feasibility, GPU throughput vs Fortran overhead, European continental vs Drava basin pilot. Need maybe compute some stats: 5000 paired runs across 1000 parcels and 5 scenarios = 5 per parcel? If paired observations are parcel-scenario pairs. Train 3500 = 70% maybe 700 parcels*5? Test 1500 = 300 parcels*5? If strictly isolated by parcel, test parcels not in train. But if split by scenario? Need audit. 1000 parcels * 5 scenarios = 5000. If train/test split 70/30 at observation level, likely same parcels appear in both train and test across scenarios, causing leakage. Need strict parcel-level split: 700 train parcels, 300 test parcels, each with 5 scenarios. Or scenario-level split? If scenarios are fixed climate scenarios, cannot split scenarios if all parcels have all scenarios? Could do leave-scenario-out for OOD. But 5 scenarios small. Need discuss.

Need evaluate sample size: 5000 paired observations. For ML surrogate, maybe adequate for interpolation if features low-dimensional and smooth, but for 1000 spatial units and 5 scenarios, effective independent units maybe 1000 parcels, not 5000, due to repeated scenarios. If spatial autocorrelation, effective sample size lower. Need power: if evaluating accuracy (MAE/RMSE) on 1500 test, CI width. Could approximate if errors normal with

## Iteration Round 2 Audit Notes
Timestamp: 2026-11-02 21:54:33

### Biophysical Auditor Evaluation (Gemma 4)
### **AUDIT REPORT: BIOPHYSICAL INVARIANT VIOLATIONS IN SURROGATE MODELS**
**TO:** Model Development Team  
**FROM:** ADVERSARIAL BIOPHYSICAL AUDITOR (G

### Quantitative Auditor Evaluation (Qwen 27B)
We need answer user's request. Need evaluate statistical validity and operational feasibility of grounded findings. Need likely produce rigorous audit. Need think through details.

User: "Evaluate the statistical validity and operational feasibility of the grounded findings:
- 5,000 paired observations (3,500 train / 1,500 test across 1,000 parcels and 5 scenarios).
- Hybrid Dispatcher achieves 93.1% GPU surrogate execution with 6.9% Fortran fallback.
- Osijek pilot site (54 parcels, 449.7 ha, 144 out of 184 days in soil water deficit).
Audit the sample size, test-set purity, and operational efficiency tradeoffs."

We need as senior quantitative econometrician/statistical methodologist. Need scrutinize sample size, train/test isolation, computational tradeoff, contrast European continental dynamics with Drava River basin pilot. Ensure conclusions robust not artifacts.

Need likely discuss:
- 5000 paired observations: paired runs across 1,000 parcels and 5 climate scenarios. If each parcel has 5 scenario runs, total 5000. Train 3500, test 1500. Need check if split is independent. If split by parcel? If random row split, leakage because same parcel appears in train and test across scenarios? Need test-set purity: must be parcel-level holdout, not scenario-level? If 1000 parcels, 5 scenarios each. To have 1500 test, could be 300 parcels x5 scenarios = 1500. Train 700 parcels x5 = 3500. That is clean if all scenarios for parcel in same set. If split by scenario? 5 scenarios, 1000 parcels each. 1500 test could be 1.5 scenarios? Not integer. If random 30% rows, likely same parcel in both train/test, causing spatial/temporal correlation leakage. Need require group-based CV or leave-parcel-out. Also climate scenarios may be correlated; if scenarios are perturbations of same parcel, test should include unseen parcels or unseen scenario-parcel combinations? For OOD validity, test should include out-of-distribution climate scenarios or spatially independent parcels. If all 5 scenarios included in training,

