"""
Score patients for in-hospital mortality risk after PCI for unprotected left main
disease, using the natural-distribution random forest (well calibrated; no recalibration).

Usage:
    python score_patients.py input.csv            # writes input_scored.csv
    python score_patients.py input.csv out.csv

Required CSV columns (0/1 for binary; numbers for the last two):
    Sex, Age_Cat70, Smoking, DM, Hx_MI, Hx_PCI, Hx_CABG, CKD, PAD, Hx_CVA, AF,
    Hx_CHF, ACS_STEMI, EF_Cat40, PulmonaryHTN, VHD, CrCl_Cat60, ProximalLM,
    Brady, Tachy, Arrest, shock, EuroScore, Angio_SYNTAX
"""
import sys, os, pickle
import numpy as np, pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pci_lm_mortality_model.pkl')

def _load():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def score(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with mortality_prob, mortality_pct, risk_category."""
    m = _load(); meta = m['meta']; feats = meta['features']; bands = meta['risk_bands']
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    prob = m['rf'].predict_proba(df[feats])[:, 1]           # raw = calibrated (natural distribution)
    lo, hi = bands['low']['max'], bands['high']['min']
    cat = np.where(prob >= hi, 'High', np.where(prob >= lo, 'Intermediate', 'Low'))
    out = df.copy()
    out['mortality_prob'] = np.round(prob, 4)
    out['mortality_pct'] = np.round(prob * 100, 1)
    out['risk_category'] = cat
    return out

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(inp)[0] + '_scored.csv'
    res = score(pd.read_csv(inp))
    res.to_csv(out, index=False)
    print(f"Scored {len(res)} patients -> {out}")
    print(res[['mortality_pct', 'risk_category']].to_string())

if __name__ == '__main__':
    main()
