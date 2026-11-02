# Chief Scientific Supervision: Iteration 2
Supervisor: Main Qwen MoE (127.0.0.1:8080)
Date: 2026-11-02 21:55:40

## Formal Editorial Review — Iteration Round 2

**Reviewer role:** Senior quantitative econometrician / statistical methodologist  
**Scope:** Statistical validity, train/test purity, operational feasibility, and physical robustness of the reported surrogate–Fortran hybrid dispatch system.  
**Grounded claims under review:**

1. **5,000 paired observations**  
   - 3,500 training observations  
   - 1,500 test observations  
   - 1,000 parcels × 5 scenarios  

2. **Hybrid Dispatcher performance**  
   - 93.1% GPU surrogate execution  
   - 6.9% Fortran fallback  

3. **Osijek pilot site**  
   - 54 parcels  
   - 449.7 ha  
   - 144 out of 184 days in soil water deficit  

4. **Additional claimed metrics to audit**  
   - 1,614 negative irrigation events  
   - 37.35 t/ha maximum erosion error  

---

# 1. Sample Size and Statistical Validity

## 1.1 Nominal sample size is adequate, but independence is the central issue

The nominal sample size is:

\[
N = 5{,}000
\]

with:

\[
N_{\text{train}} = 3{,}500
\]

\[
N_{\text{test}} = 1{,}500
\]

This is a reasonably large sample for a surrogate-model evaluation, especially if the observations are paired surrogate–Fortran runs. However, the statistical validity depends less on the raw count than on the **dependence structure** of the data.

The data are not 5,000 independent observations. They are structured as:

\[
1{,}000 \text{ parcels} \times 5 \text{ scenarios}
\]

Therefore, the effective sample size is not 5,000 independent draws. It is closer to 1,000 spatial units with 5 correlated scenario repetitions per parcel.

This matters because parcels likely share:

- soil type,
- topography,
- land use,
- management history,
- local climate,
- hydrological connectivity
