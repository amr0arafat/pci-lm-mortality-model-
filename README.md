# Machine-learning prediction of in-hospital mortality after PCI for unprotected left main disease

Reproducible code, trained model, and an interactive risk calculator accompanying the manuscript
*"Development and explainability of a machine-learning model for predicting in-hospital mortality
after percutaneous coronary intervention for unprotected left main coronary artery disease"*
(Gulf-LM registry).

The final model is a **random forest trained on the natural class distribution** (no oversampling or
class weighting), which is well calibrated without recalibration.

| Metric (out-of-fold) | Value |
|---|---|
| AUROC | 0.904 (bootstrap 95% CI 0.856–0.940) |
| AUPRC | 0.353 |
| Brier score | 0.029 |
| Calibration-in-the-large | ≈ 0 (mean predicted 3.7% vs observed 3.7%) |
| Calibration slope | 1.20 |

Observed in-hospital mortality by predicted-risk band: **< 2% → 0.3%**, **2–5% → 4.2%**, **≥ 5% → 15.1%**.

## Interactive risk calculator

Open **[`docs/index.html`](docs/index.html)** in any web browser — it is fully self-contained
(the model is embedded), runs offline, and needs no installation. When this repository is published
with GitHub Pages enabled, the calculator is served live at
`https://<your-username>.github.io/pci-lm-mortality-model/`.

## Repository contents

```
model/     pci_lm_mortality_model.pkl   trained random forest + metadata
           model_web.json               model exported for the browser calculator
           model_card.json              performance, risk bands, feature list
           score_patients.py            CLI: python score_patients.py patients.csv
           app.py                       optional Streamlit calculator
docs/      index.html                   self-contained offline risk calculator (served by Pages)
analysis/  PCI_LM_mortality_FINAL_analysis.py / .ipynb / .html   full reproducible analysis
figures/   manuscript figures (PNG + SVG)
```

## Reproducing the analysis

```bash
pip install -r requirements.txt
# place the registry data (PCIdata.xls, sheet.dta) beside the notebook, then:
jupyter notebook analysis/PCI_LM_mortality_FINAL_analysis.ipynb
```

The notebook covers data preparation and leakage control, the model library, out-of-fold
evaluation with bootstrap confidence intervals (Table 2), calibration and risk bands, SHAP and
permutation importance, subgroup/fairness analyses, and the sensitivity analyses (prior-CABG
exclusion, a parsimonious model, and multiple imputation).

## Scoring new patients

```bash
python model/score_patients.py patients.csv   # -> patients_scored.csv (probability + risk band)
```

Required columns: `Sex, Age_Cat70, Smoking, DM, Hx_MI, Hx_PCI, Hx_CABG, CKD, PAD, Hx_CVA, AF,
Hx_CHF, ACS_STEMI, EF_Cat40, PulmonaryHTN, VHD, CrCl_Cat60, ProximalLM, Brady, Tachy, Arrest, shock`
(0/1), and `EuroScore`, `Angio_SYNTAX` (numeric).

## Data availability

The **Gulf-LM registry patient-level data are not included** in this repository. They are available
from the corresponding author on reasonable request, subject to the registry's data-governance and
ethical-approval requirements. Only the code, the trained model, and the calculator are shared here.

## Intended use and limitations

This is a research tool developed with **internal validation only** in a single registry with 45
outcome events. It is **not** a substitute for clinical judgement and must not be used for clinical
decisions without prospective external validation. Its principal value is calibrated risk
stratification and a high negative predictive value (ruling out death), rather than confirming high
risk, where the positive predictive value is limited by the low event rate.

## Citation

See `CITATION.cff`. If you archive a release on Zenodo, cite the resulting DOI.

## License

Code and model: MIT License (`LICENSE`). Registry data are not covered by this license and are not distributed.
