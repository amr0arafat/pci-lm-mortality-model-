"""
Streamlit risk calculator — in-hospital mortality after PCI for unprotected left main disease.
Natural-distribution random forest (well calibrated; no recalibration).
Run:  pip install streamlit scikit-learn pandas numpy  &&  streamlit run app.py
"""
import os, pickle
import numpy as np, pandas as pd, streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pci_lm_mortality_model.pkl')

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

m = load_model(); meta = m['meta']
LABELS = meta['labels']; bands = meta['risk_bands']
lo, hi = bands['low']['max'], bands['high']['min']

st.set_page_config(page_title='PCI Left-Main Mortality Risk', page_icon='❤️', layout='centered')
st.title('In-hospital mortality risk after PCI for unprotected left main disease')
st.caption('Random forest trained on the natural class distribution (well calibrated). '
           'Gulf-LM registry (n=1,205; 45 deaths). Out-of-fold AUROC %.3f. '
           'Research tool — not for clinical use without external validation.' % meta['performance']['auroc'])

groups = {'Demographics':['Sex','Age_Cat70','Smoking'],
          'Comorbidities':['DM','Hx_MI','Hx_PCI','Hx_CABG','CKD','PAD','Hx_CVA','AF','Hx_CHF'],
          'Presentation':['ACS_STEMI','Brady','Tachy','Arrest','shock'],
          'Investigations':['EF_Cat40','PulmonaryHTN','VHD','CrCl_Cat60','ProximalLM']}
vals = {}; cols = st.columns(2); i = 0
for gname, feats in groups.items():
    with cols[i % 2]:
        st.markdown(f"**{gname}**")
        for f in feats:
            vals[f] = 1 if st.checkbox(LABELS[f], key=f) else 0
    i += 1
c1, c2 = st.columns(2)
vals['EuroScore'] = c1.number_input('EuroSCORE II (%)', 0.0, 100.0, 3.6, 0.1)
vals['Angio_SYNTAX'] = c2.number_input('SYNTAX score', 0.0, 100.0, 28.0, 1.0)

if st.button('Calculate risk', type='primary'):
    row = pd.DataFrame([[vals[f] for f in meta['features']]], columns=meta['features'])
    prob = float(m['rf'].predict_proba(row)[:, 1][0])
    if prob >= hi:
        cat, color, obs = 'HIGH risk', '#c0392b', bands['high']['observed']
    elif prob >= lo:
        cat, color, obs = 'INTERMEDIATE risk', '#e67e22', bands['intermediate']['observed']
    else:
        cat, color, obs = 'LOW risk', '#27ae60', bands['low']['observed']
    st.markdown(f"<h2 style='color:{color}'>Predicted mortality: {prob*100:.1f}% &nbsp;—&nbsp; {cat}</h2>", unsafe_allow_html=True)
    st.progress(min(prob / max(hi*2, 0.1), 1.0))
    st.caption(f"Risk bands: Low < {lo*100:.0f}% ≤ Intermediate < {hi*100:.0f}% ≤ High. "
               f"Observed in-hospital mortality in this band in the derivation cohort: {obs*100:.1f}%.")
