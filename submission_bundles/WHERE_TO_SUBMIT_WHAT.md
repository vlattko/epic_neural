# Master Submission Guide: Where to Submit What

This directory contains pre-packaged, verified bundles prepared for each of your submission targets.

---

## Directory Organization Overview

```text
D:\Kaggle\submission_bundles\
├── WHERE_TO_SUBMIT_WHAT.md                 <-- Master Guide (You Are Here)
│
├── 01_kaggle_submission\                  <-- BUNDLE 1: KAGGLE COMPETITION
│   ├── paper_submission.md                 (Polished paper with zero em dashes)
│   ├── kaggle_metadata_and_form.md         (Ready-to-copy fields: Title, Abstract, Links)
│   ├── kaggle_quick_verification.py        (1-command test & benchmark script for judges)
│   └── figures\                            (All 8 publication-grade 300 DPI figures)
│
├── 02_arxiv_preprint\                      <-- BUNDLE 2: ARXIV / EARTHARXIV PREPRINT
│   ├── arxiv_upload_package.zip            (Pre-packaged ZIP ready for direct upload to arXiv.org)
│   ├── manuscript.tex                      (Complete, standalone LaTeX source with figures & citations)
│   ├── manuscript.md                       (Full manuscript formatted for markdown preprint servers)
│   ├── arxiv_submission_metadata.txt       (Categories, Comments, MSC/ACM classes, License)
│   ├── build_pdf.bat                       (1-click script to compile PDF via Pandoc/Typst)
│   └── figures\                            (High-res figures for TeX/PDF compilation)
│
└── 03_github_release_bundle\               <-- BUNDLE 3: OPEN-SOURCE GITHUB REPO
    ├── README.md                           (Public-facing repo documentation with badges)
    └── RELEASE_NOTES_v1.0.md               (Formal v1.0.0-grounded release notes)
```

---

## Target 1: Kaggle Gemma 4 Developer Agent Paper Track

- **Destination URL:** `https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview`
- **Submission Window / Format:** Upload paper (PDF or Markdown) + Link to Public Codebase.

### Step-by-Step Submission Instructions:
1. Open the [Kaggle Competition Page](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview) and click **"Submit"** or create a new public notebook/writeup.
2. Open [`01_kaggle_submission/kaggle_metadata_and_form.md`](01_kaggle_submission/kaggle_metadata_and_form.md):
   - Copy the **Title**: `Autonomous Multi-Agent Scientific Discovery with Long-Context Gemma 4 and Decentralized Heterogeneous Computing: Unraveling the European Soil-Water Nexus`.
   - Copy the **Subtitle** and **Abstract** directly into the submission fields.
3. Upload the paper text:
   - Provide [`01_kaggle_submission/paper_submission.md`](01_kaggle_submission/paper_submission.md) (or compile it to PDF using `build_pdf.bat` and upload `paper_submission.pdf`).
4. Link the Codebase:
   - Provide your GitHub repository URL: `https://github.com/vlattko/soil-water-nexus-agents`.
   - Mention the one-click judge verification: *"Judges can verify all 76 unit tests and empirical ledgers with `python kaggle_quick_verification.py`"*.
5. Attach key figures from [`01_kaggle_submission/figures/`](01_kaggle_submission/figures/):
   - `fig1_decentralized_cluster_dag.png` (Hardware topology)
   - `fig1_grounded_failure_regimes.png` (Surrogate failure physics)
   - `fig2_grounded_hybrid_guardrail.png` (Hybrid dispatch guardrail)
   - `fig3_grounded_pilot_drava.png` (In-situ Drava basin series)

---

## Target 2: arXiv / EarthArXiv Preprint

- **Option A (arXiv):** `https://arxiv.org/submit`
  - Recommended for Computer Science / AI developer agent framing.
  - Primary Category: `cs.AI` (Artificial Intelligence)
  - Cross Lists: `cs.SE` (Software Engineering), `q-bio.QM` (Quantitative Methods), `stat.ML` (Machine Learning)
- **Option B (EarthArXiv / bioRxiv):** `https://eartharxiv.org/` or `https://www.biorxiv.org/`
  - Recommended for instant publication without needing an arXiv author endorsement.
  - Categories: Environmental Sciences / Computational Agronomy / Pedology.

### Step-by-Step Submission Instructions (arXiv TeX Direct Upload):
1. Navigate to [arXiv Submission Portal](https://arxiv.org/submit) and log in.
2. Select Primary Category: `cs.AI` (Artificial Intelligence).
   - Cross Lists: `cs.SE` (Software Engineering), `q-bio.QM` (Quantitative Methods), `stat.ML` (Machine Learning).
3. Open [`02_arxiv_preprint/arxiv_submission_metadata.txt`](02_arxiv_preprint/arxiv_submission_metadata.txt):
   - Copy the Title, Author list, TeX-compatible Abstract, and MSC/ACM classification codes into the web form.
4. Upload Files:
   - Simply upload [`02_arxiv_preprint/arxiv_upload_package.zip`](02_arxiv_preprint/arxiv_upload_package.zip).
   - It contains `manuscript.tex` at the root and all required figures in `figures/`.
   - Click "Process Files" — arXiv AutoTeX will automatically compile it to PDF.
5. Review the AutoTeX generated PDF preview, verify that all 4 figures render crisply, and choose **CC-BY 4.0** license.
6. Submit! arXiv will issue an identifier (e.g., `arXiv:2611.XXXXX`).
7. Update the subtitle of your Kaggle submission with this preprint link.

---

## Target 3: GitHub Open-Source Release

- **Destination URL:** `https://github.com/vlattko/soil-water-nexus-agents`

### Step-by-Step Release Instructions:
1. Push local changes from `D:\Kaggle` to GitHub:
   ```bash
   git push origin master
   ```
2. Navigate to GitHub -> **Releases** -> **"Draft a new release"**.
3. Set Tag: `v1.0.0-grounded`.
4. Release Title: `v1.0.0 Grounded Multi-Agent Discovery & Hybrid Dispatcher`.
5. Copy contents of [`03_github_release_bundle/RELEASE_NOTES_v1.0.md`](03_github_release_bundle/RELEASE_NOTES_v1.0.md) into the description.
6. Click **"Publish release"**.

---

## Summary Checklist

| Destination | Bundle Path | Key File to Submit | Action Status |
| :--- | :--- | :--- | :--- |
| **Kaggle Track** | `01_kaggle_submission/` | `paper_submission.md` / `kaggle_metadata_and_form.md` | **Ready to Submit** |
| **arXiv / EarthArXiv** | `02_arxiv_preprint/` | `manuscript.md` / `arxiv_submission_metadata.txt` | **Ready to Submit** |
| **GitHub Release** | `03_github_release_bundle/` | `README.md` / `RELEASE_NOTES_v1.0.md` | **Ready to Push** |
