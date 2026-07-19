# How to publish this package (step-by-step)

Everything below uses only free services. Choose the paths you need.

## A. For peer review (anonymous — do this first)

The simplest, fully anonymous option needs no external accounts:

1. Zip this folder (a ready-made `pci-lm-mortality-model_supplementary.zip` is provided).
2. Upload it as a **Supplementary File** in the journal's submission system.
3. Also upload `docs/index.html` on its own so reviewers can open the **calculator** directly.

If you want a *live* anonymous link, paste your GitHub repo URL into
**https://anonymous.4open.science** (it serves an anonymized mirror), or create an **OSF**
project and share an **anonymized view-only link**. Do **not** put a personal GitHub/Streamlit URL
in a blinded manuscript.

## B. Publish the code on GitHub (for readers)

```bash
cd pci-lm-mortality-model
git init && git add . && git commit -m "Initial release: model, code, calculator"
# create an empty repo named pci-lm-mortality-model on github.com, then:
git branch -M main
git remote add origin https://github.com/<your-username>/pci-lm-mortality-model.git
git push -u origin main
```
The `.gitignore` already prevents any patient data (`*.xls`, `*.dta`, `*.csv`) from being committed.

## C. Host the interactive calculator on GitHub Pages (free, permanent)

1. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. The included workflow (`.github/workflows/pages.yml`) publishes `docs/index.html` automatically on
   every push to `main`.
3. Your calculator goes live at:
   `https://<your-username>.github.io/pci-lm-mortality-model/`
4. Put that URL in the manuscript's Code/Data Availability statement (after acceptance).

## D. Mint a citable DOI on Zenodo (do this at acceptance)

1. Sign in to **https://zenodo.org** with your GitHub account.
2. **Zenodo → GitHub**, flip the switch **ON** for `pci-lm-mortality-model`.
3. On GitHub, create a **Release** (e.g., tag `v1.0.0`). Zenodo automatically archives it and issues a **DOI**.
4. Add the DOI to `CITATION.cff` and to the manuscript.

## E. (Optional) Host the richer Streamlit app

**Streamlit Community Cloud:** https://share.streamlit.io → New app → point to
`model/app.py` in your repo. (Requires `requirements.txt`, already included.)

**Hugging Face Spaces:** create a Space (SDK: Streamlit), upload `model/app.py`,
`model/pci_lm_mortality_model.pkl`, and `requirements.txt`.

## Suggested Code/Data Availability statement for the manuscript

> The analysis code, the trained model, and an interactive risk calculator are available at
> https://github.com/<user>/pci-lm-mortality-model (archived at Zenodo, doi:10.5281/zenodo.XXXXXXX);
> the calculator can be used at https://<user>.github.io/pci-lm-mortality-model/. The Gulf-LM
> registry patient-level data are available from the corresponding author on reasonable request,
> subject to registry governance and ethical approval.
